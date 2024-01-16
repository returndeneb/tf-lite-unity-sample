using TensorFlowLite;
using UnityEngine;

namespace Holistic
{
    public class FaceMesh : BaseImagePredictor<float>
    {
        public class Result
        {
            public float score;
            public Vector3[] keyPoints;
        }

        private const int KeypointCount = 468;
        private readonly float[,] output0 = new float[KeypointCount, 3]; // keypoint
        private readonly float[] output1 = new float[1]; // flag

        private readonly Result result;
        private Matrix4x4 cropMatrix;

        private Vector2 FaceShift { get; set; } = new Vector2(0f, 0f);
        private Vector2 FaceScale { get; set; } = new Vector2(1.6f, 1.6f);
        public Matrix4x4 CropMatrix => cropMatrix;


        public FaceMesh(string modelPath) : base(modelPath, Accelerator.GPU)
        {
            result = new Result()
            {
                score = 0,
                keyPoints = new Vector3[KeypointCount],
            };
        }

        public override void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture inputTex, FaceDetect.Result palm)");
        }

        public void Invoke(Texture inputTex, FaceDetect.Result face)
        {
            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = face.rect,
                rotationDegree = CalcFaceRotation(ref face) * Mathf.Rad2Deg,
                shift = FaceShift,
                scale = FaceScale,
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
        }

        public Result GetResult()
        {
            const float SCALE = 1f / 192f;
            var mtx = cropMatrix.inverse;

            result.score = output1[0];
            for (var i = 0; i < KeypointCount; i++)
            {
                result.keyPoints[i] = mtx.MultiplyPoint3x4(new Vector3(
                    output0[i, 0] * SCALE,
                    1f - output0[i, 1] * SCALE,
                    output0[i, 2] * SCALE
                ));
            }
            return result;
        }

        public FaceDetect.Result LandmarkToDetection(Result landmark)
        {
            // Original index looks like a bug
            // rotation_vector_start_keypoint_index: 33  # Left side of left eye.
            // rotation_vector_end_keypoint_index: 133  # Right side of right eye.

            const int start = 33; // Left side of left eye.
            const int end = 263; // Right side of right eye.

            var landmarkKeyPoints = landmark.keyPoints;
            for (var i = 0; i < landmarkKeyPoints.Length; i++)
            {
                var v = landmarkKeyPoints[i];
                v.y = 1f - v.y;
                landmarkKeyPoints[i] = v;
            }

            var rect = RectExtension.GetBoundingBox(landmarkKeyPoints);
            var center = rect.center;
            var size = Mathf.Min(rect.width, rect.height);
            rect = new Rect(center.x - size * 0.5f, center.y - size * 0.5f, size, size);

            return new FaceDetect.Result()
            {
                score = landmark.score,
                rect = rect,
                keyPoints = new Vector2[]
                {
                    landmarkKeyPoints[end], landmarkKeyPoints[start]
                },
            };
        }

        private static float CalcFaceRotation(ref FaceDetect.Result detection)
        {
            var vec = detection.RightEye - detection.LeftEye;
            return -Mathf.Atan2(vec.y, vec.x);
        }
    }
}
