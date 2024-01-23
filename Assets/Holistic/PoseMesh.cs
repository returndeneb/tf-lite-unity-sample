using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using UnityEngine;

namespace Holistic
{
    public sealed class PoseMesh : ImageInterpreter<float>
    {
        public class Result
        {
            public float score;
            public Vector4[] viewportLandmarks;
            public Vector4[] worldLandmarks;
        }

        public const int LandmarkCount = 33;
        // A pair of indexes
        public static readonly int[] Connections = {
            // the same as Upper Body 
            0, 1, 1, 2, 2, 3, 3, 7, 0, 4, 4, 5, 5, 6, 6, 8, 9, 10, 11, 12, 11, 13, 13, 15, 15, 17, 15, 19, 15, 21, 
            17, 19, 12, 14, 14, 16, 16, 18, 16, 20, 16, 22, 18, 20, 11, 23, 12, 24, 23, 24,
            // left leg
            24, 26, 26, 28, 28, 32, 32, 30, 30, 28,
            // right leg
            23, 25, 25, 27, 27, 31, 31, 29, 29, 27,
        };

        // ld_3d
        private readonly float[] output0 = new float[195];
        // output_poseflag
        private readonly float[] output1 = new float[1];
        // output_segmentation
        private readonly float[,] output2 = new float[256, 256];
        // output_heatmap; not in use
        // private readonly float[,,] output3 = new float[64, 64, 39];
        // world_3d
        private readonly float[] output4 = new float[117];

        private readonly Result result;
        private readonly RelativeVelocityFilter3D[] filters;
        // private readonly Options options;
        private readonly PoseSegmentation segmentation;
        private Matrix4x4 cropMatrix;

        public Matrix4x4 CropMatrix => cropMatrix;

        public PoseMesh(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            // this.options = options;
            // resizeOptions.aspectMode = options.AspectMode;

            result = new Result()
            {
                score = 0,
                viewportLandmarks = new Vector4[LandmarkCount],
                worldLandmarks = null
            };

          

            // Init filters
            filters = new RelativeVelocityFilter3D[LandmarkCount];
            const int windowSize = 5;
            const float velocityScale = 10;
            const RelativeVelocityFilter.DistanceEstimationMode mode =
                RelativeVelocityFilter.DistanceEstimationMode.LegacyTransition;
            for (var i = 0; i < LandmarkCount; i++)
            {
                filters[i] = new RelativeVelocityFilter3D(windowSize, velocityScale, mode);
            }
        }

        public override void Dispose()
        {
            segmentation?.Dispose();
            base.Dispose();
        }

        public void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture, PalmDetect.Result)");
        }

        public Result Invoke(Texture inputTex, PoseDetect.Result pose)
        {
            cropMatrix = CalcCropMatrix(ref pose, ref resizeOptions);

            var rt = resizer.Resize(
               inputTex, resizeOptions.width, resizeOptions.height,
               cropMatrix,
               TextureResizer.GetTextureST(inputTex, resizeOptions));
            ToTensor(rt, inputTensor, false);

            InvokeInternal();

            return GetResult(inputTex);
        }

        public async UniTask<Result> InvokeAsync(Texture inputTex, PoseDetect.Result pose,
            CancellationToken cancellationToken, PlayerLoopTiming timing)
        {
            cropMatrix = CalcCropMatrix(ref pose, ref resizeOptions);
            var rt = resizer.Resize(
              inputTex, resizeOptions.width, resizeOptions.height,
              cropMatrix,
              TextureResizer.GetTextureST(inputTex, resizeOptions));
            await ToTensorAsync(rt, inputTensor, false, cancellationToken);
            await UniTask.SwitchToThreadPool();

            InvokeInternal();

            await UniTask.SwitchToMainThread(timing, cancellationToken);
            return GetResult(inputTex);
        }

        private void InvokeInternal()
        {
            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
            
        }

        private Result GetResult(Texture inputTex)
        {
            // Normalize 0 ~ 255 => 0.0 ~ 1.0
            const float scale = 1f / 255f;
            var mtx = cropMatrix.inverse;

            // https://google.github.io/mediapipe/solutions/pose.html#output
            // The magnitude of z uses roughly the same scale as x.
            var xScale = Mathf.Abs(mtx.lossyScale.x);
            var zScale = scale * xScale * xScale;

            result.score = output1[0];

            var min = new Vector2(float.MaxValue, float.MaxValue);
            var max = new Vector2(float.MinValue, float.MinValue);

            var dimensions = output0.Length / LandmarkCount;

            for (var i = 0; i < LandmarkCount; i++)
            {
                Vector4 p = mtx.MultiplyPoint3x4(new Vector3(
                    output0[i * dimensions] * scale,
                    1f - output0[i * dimensions + 1] * scale,
                    output0[i * dimensions + 2] * zScale
                ));
                p.w = output0[i * dimensions + 3];
                result.viewportLandmarks[i] = p;

                if (p.x < min.x) { min.x = p.x; }
                if (p.x > max.x) { max.x = p.x; }
                if (p.y < min.y) { min.y = p.y; }
                if (p.y > max.y) { max.y = p.y; }
            }
            

            return result;
        }

        private void SetWorldLandmarks()
        {
            var dimensions = output4.Length / LandmarkCount;
            for (var i = 0; i < LandmarkCount; i++)
            {
                result.worldLandmarks[i] = new Vector4(
                    output4[i * dimensions],
                    -output4[i * dimensions + 1],
                    output4[i * dimensions + 2],
                    result.viewportLandmarks[i].w
                );
            }
        }

        private static Rect AlignmentPointsToRect(in Vector2 center, in Vector2 scale)
        {
            float boxSize = Mathf.Sqrt(
                (scale.x - center.x) * (scale.x - center.x)
                + (scale.y - center.y) * (scale.y - center.y)
            ) * 2f;
            return new Rect(
                center.x - boxSize / 2,
                center.y - boxSize / 2,
                boxSize,
                boxSize);
        }

        private static float CalcRotationDegree(in Vector2 a, in Vector2 b)
        {
            var vec = a - b;
            return  -Mathf.Atan2(vec.y, vec.x) * Mathf.Rad2Deg - 90f;
        }

        private void UpdateFilterScale(Vector3 scale)
        {
            foreach (var f in filters)
            {
                f.VelocityScale = scale;
            }
        }

        private static Matrix4x4 CalcCropMatrix(ref PoseDetect.Result pose, ref TextureResizer.ResizeOptions resizeOptions)
        {
            var rotation = CalcRotationDegree(pose.keypoints[0], pose.keypoints[1]);
            var rect = AlignmentPointsToRect(pose.keypoints[0], pose.keypoints[1]);
            return RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = rect,
                rotationDegree = rotation,
                shift = new Vector2(0, 0),
                scale = new Vector2(1.5f, 1.5f),
                mirrorHorizontal = resizeOptions.mirrorHorizontal,
                mirrorVertical = resizeOptions.mirrorVertical,
            });
        }

        public static PoseDetect.Result LandmarkToDetection(Result result)
        {
            Vector2 hip = (result.viewportLandmarks[24] + result.viewportLandmarks[23]) / 2f;
            Vector2 nose = result.viewportLandmarks[0];
            var aboveHead = hip + (nose - hip) * 1.2f;
            // Y Flipping
            hip.y = 1f - hip.y;
            aboveHead.y = 1f - aboveHead.y;

            return new PoseDetect.Result()
            {
                score = result.score,
                keypoints = new[] { hip, aboveHead }
            };
        }
    }
}
