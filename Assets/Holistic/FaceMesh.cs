using TensorFlowLite;
using UnityEngine;

namespace Holistic
{
    public class FaceMesh : ImageInterpreter<float>
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

        private Vector2 FaceShift { get; set; } = new(0f, 0f);
        private Vector2 FaceScale { get; set; } = new(1f, 1f);
        
        public FaceMesh(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            result = new Result()
            {
                score = 0,
                keyPoints = new Vector3[KeypointCount],
            };
        }

        public virtual void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture inputTex, FaceDetect.Result palm)");
        }

        public void Invoke(Texture inputTex, FaceDetect.Result face)
        {
            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = face.rect,
                rotationDegree = face.rotation,
                // shift = FaceShift,
                scale = FaceScale,
                // mirrorHorizontal = resizeOptions.mirrorHorizontal,
                // mirrorVertical = resizeOptions.mirrorVertical,
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
            const float scale = 1f / 192f;
            var mtx = cropMatrix.inverse;

            result.score = output1[0];
            for (var i = 0; i < KeypointCount; i++)
            {
                result.keyPoints[i] = mtx.MultiplyPoint3x4(new Vector3(
                    output0[i, 0] * scale,
                    1f - output0[i, 1] * scale,
                    output0[i, 2] * scale
                ));
            }
            return result;
        }

        public static FaceDetect.Result LandmarkToDetection(Result landmark)
        {
            const int start = 33; // Left side of left eye.
            const int end = 263; // Right side of right eye.

            var landmarkKeyPoints = landmark.keyPoints;
            
            for (var i = 0; i < landmarkKeyPoints.Length; i++)
            {
                landmarkKeyPoints[i].y = 1f - landmarkKeyPoints[i].y;
            }

            var rect = RectExtension.GetBoundingBox(landmarkKeyPoints);
            var center = rect.center;
            var size = Mathf.Min(rect.width, rect.height)*1.6f;
            
            var vec =  landmarkKeyPoints[end] - landmarkKeyPoints[start];
            
            return new FaceDetect.Result
            {
                score = landmark.score,
                rect = new Rect(center.x - size * 0.5f, center.y - size * 0.5f, size, size),
                rotation = -Mathf.Atan2(vec.y, vec.x)*Mathf.Rad2Deg
            };
        }
    }
}
