using TensorFlowLite;
using UnityEngine;

namespace Holistic
{
    public class FaceMesh : ImageInterpreter<float>
    {
        public class Result
        {
            public Vector3[] keyPoints;
            public float score;
            
        }

        private const int KeypointCount = 468;
        private readonly float[,] output0 = new float[KeypointCount, 3]; // key points
        private readonly float[] output1 = new float[1]; // score

        private readonly Result result;
        private Matrix4x4 cropMatrix;

        private Vector2 FaceShift { get; set; } = new(0f, 0f);
        private Vector2 FaceScale { get; set; } = new(1.6f, 1.6f);
        
        public FaceMesh(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            result = new Result()
            {
                keyPoints = new Vector3[KeypointCount],
                score = 0,
            };
        }
        

        public void Invoke(Texture inputTex, FaceDetect.Result face)
        {
            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options
            {
                rect = face.rect,
                rotationDegree = face.rotation,
                // shift = FaceShift,
                scale = FaceScale,
            });
            
            var rt = resizer.Resize(
                inputTex, resizeOptions.width, resizeOptions.height,
                cropMatrix,
                TextureResizer.GetTextureSt(inputTex, resizeOptions));
            
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
                    1-output0[i, 1] * scale,
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
            var size = Mathf.Min(rect.width, rect.height);
            rect = new Rect(center.x - size * 0.5f, center.y - size * 0.5f, size, size);

            var vec =  landmarkKeyPoints[end] - landmarkKeyPoints[start];
            
            return new FaceDetect.Result
            {
                score = landmark.score,
                rect = rect,
                rotation = -Mathf.Atan2(vec.y, vec.x)*Mathf.Rad2Deg
            };
        }
    }
}
