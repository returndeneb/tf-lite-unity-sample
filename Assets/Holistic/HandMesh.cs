using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.Rendering;

namespace Holistic
{
    public class HandMesh : ImageInterpreter<float>
    {
        public class Result
        {
            public float score;
            public Vector3[] joints;
        }

        public enum Dimension
        {
            Two,
            Three,
        }

        public static readonly int[] Connections = new int[] { 0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 5, 9, 9,
            10, 10, 11, 11, 12, 9, 13, 13, 14, 14, 15, 15, 16, 13, 17, 0, 17, 17, 18, 18, 19, 19, 20, };
        public const int JointCount = 21;

        private readonly float[] output0; // keypoints
        private readonly float[] output1 = new float[1]; // score
        private readonly float[] output2 = new float[1]; // 오른손 왼손
        
        private readonly Result result;
        private Matrix4x4 cropMatrix;

        private Dimension Dim { get; }
        private Vector2 PalmShift { get; } = new Vector2(0, 0.2f);
        private Vector2 PalmScale { get; } = new Vector2(2.8f, 2.8f);
        public Matrix4x4 CropMatrix => cropMatrix;

        public HandMesh(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            var out0Info = interpreter.GetOutputTensorInfo(0);
            Dim = out0Info.shape[1] switch
            {
                JointCount * 2 => Dimension.Two,
                JointCount * 3 => Dimension.Three,
                _ => throw new System.NotSupportedException()
            };
            output0 = new float[out0Info.shape[1]];

            result = new Result()
            {
                score = 0,
                joints = new Vector3[JointCount],
            };
        }

        public override void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture inputTex, PalmDetect.Palm palm)");
        }

        public void Invoke(Texture inputTex, HandDetect.Result palm)
        {
            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = palm.rect,
                rotationDegree = CalcHandRotation(palm),
                shift = PalmShift,
                scale = PalmScale,
                mirrorHorizontal = resizeOptions.mirrorHorizontal,
                mirrorVertical = resizeOptions.mirrorVertical,
            });

            var rt = resizer.Resize(
                inputTex, resizeOptions.width, resizeOptions.height,
                cropMatrix,
                TextureResizer.GetTextureST(inputTex, resizeOptions));
            ToTensor(rt, inputTensor, false);

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
            interpreter.GetOutputTensorData(2, output2);
        }

        public async UniTask<Result> InvokeAsync(Texture inputTex, HandDetect.Result palm, CancellationToken cancellationToken)
        {
            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = palm.rect,
                rotationDegree = CalcHandRotation(palm),
                shift = PalmShift,
                scale = PalmScale,
                mirrorHorizontal = resizeOptions.mirrorHorizontal,
                mirrorVertical = resizeOptions.mirrorVertical,
            });

            var rt = resizer.Resize(
                inputTex, resizeOptions.width, resizeOptions.height,
                cropMatrix,
                TextureResizer.GetTextureST(inputTex, resizeOptions));
            await ToTensorAsync(rt, inputTensor, false, cancellationToken);
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
            interpreter.GetOutputTensorData(2, output2);

            GetResult();
            await UniTask.SwitchToMainThread(cancellationToken);
            return result;
        }

        public Result GetResult()
        {
            // Normalize 0 ~ 255 => 0.0 ~ 1.0
            const float scale = 1f / 255f;
            var mtx = cropMatrix.inverse;

            result.score = output1[0];
            if (Dim == Dimension.Two)
            {
                for (var i = 0; i < JointCount; i++)
                {
                    result.joints[i] = mtx.MultiplyPoint3x4(new Vector3(
                        output0[i * 2] * scale,
                        1f - output0[i * 2 + 1] * scale,
                        0
                    ));
                }
            }
            else
            {
                for (var i = 0; i < JointCount; i++)
                {
                    result.joints[i] = mtx.MultiplyPoint3x4(new Vector3(
                        output0[i * 3] * scale,
                        1f - output0[i * 3 + 1] * scale,
                        output0[i * 3 + 2] * scale
                    ));
                }
            }
            return result;
        }

        private static float CalcHandRotation(HandDetect.Result detection)
        {
            // Rotation based on Center of wrist - Middle finger
            var vec = detection.keyPoints[0] - detection.keyPoints[2];
            return -90f - Mathf.Atan2(vec.y, vec.x)* Mathf.Rad2Deg;
        }
    }
}
