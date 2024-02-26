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
            public Vector3[] keyPoints;
            public float handiness;
        }


        public static readonly int[] Connections = new int[] { 0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 5, 9, 9,
            10, 10, 11, 11, 12, 9, 13, 13, 14, 14, 15, 15, 16, 13, 17, 0, 17, 17, 18, 18, 19, 19, 20, };
        public const int JointCount = 21;

        private readonly float[] keyPoints;
        private readonly float[] score = new float[1]; 
        private readonly float[] handiness = new float[1]; 
        
        private readonly Result result;
        private Matrix4x4 cropMatrix;

        private Vector2 PalmShift { get; } = new (0f, 0f);
        private Vector2 PalmScale { get; } = new (1f, 1f);
        public Matrix4x4 CropMatrix => cropMatrix;

        public HandMesh(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            var out0Info = interpreter.GetOutputTensorInfo(0);
            keyPoints = new float[out0Info.shape[1]];

            result = new Result()
            {
                score = 0,
                keyPoints = new Vector3[JointCount],
            };
        }

        public void Invoke(Texture inputTex, HandDetect.Result palm)
        {
            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = palm.rect,
                rotationDegree = palm.rotation,
                shift = PalmShift,
                scale = PalmScale,
            });

            var rt = resizer.Resize(
                inputTex, resizeOptions.width, resizeOptions.height,
                cropMatrix,
                TextureResizer.GetTextureSt(inputTex, resizeOptions));
            ToTensor(rt, inputTensor, false);

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, keyPoints);
            interpreter.GetOutputTensorData(1, score);
            interpreter.GetOutputTensorData(2, handiness);
        }

        public async UniTask<Result> InvokeAsync(Texture inputTex, HandDetect.Result palm, CancellationToken cancellationToken)
        {
            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options()
            {
                rect = palm.rect,
                rotationDegree = palm.rotation,
                shift = PalmShift,
                scale = PalmScale,
            });

            var rt = resizer.Resize(
                inputTex, resizeOptions.width, resizeOptions.height,
                cropMatrix,
                TextureResizer.GetTextureSt(inputTex, resizeOptions));
            await ToTensorAsync(rt, inputTensor, false, cancellationToken);
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, keyPoints);
            interpreter.GetOutputTensorData(1, score);
            interpreter.GetOutputTensorData(2, handiness);

            GetResult();
            await UniTask.SwitchToMainThread(cancellationToken);
            return result;
        }

        public Result GetResult()
        {
            const float scale = 255f;
            var mtx = cropMatrix.inverse;
            
            result.score = score[0];
            
            for (var i = 0; i < JointCount; i++)
            {
                result.keyPoints[i] = mtx.MultiplyPoint3x4(
                    new Vector3(keyPoints[i * 3], 
                    scale - keyPoints[i * 3 + 1], 
                    keyPoints[i * 3 + 2])/scale);
            }
            result.handiness = handiness[0];
            return result;
        }
        public static HandDetect.Result LandmarkToDetection(Result landmark)
                {
        
                    var landmarkKeyPoints = landmark.keyPoints;
                    int[] selectedIndices = { 0,1,2,3, 5,6, 9,10, 13,14, 17,18 };
                    
                    var rect = RectExtension.GetBoundingBox(landmarkKeyPoints,selectedIndices);
                    // var rect = RectExtension.GetBoundingBox(landmarkKeyPoints);
                    var center = rect.center;

                    var size = Mathf.Max(rect.width, rect.height)*2f;
                    
                    var vec =  landmarkKeyPoints[0] - landmarkKeyPoints[9];
                    var rot = -90f - Mathf.Atan2(vec.y, vec.x) * Mathf.Rad2Deg;
                    const float shifting = 0.01f;

                    return new HandDetect.Result()
                    {
                        score = landmark.score,
                        rect = new Rect(center.x + shifting*Mathf.Sin(rot*Mathf.PI/180f)- size * 0.5f, center.y + 
                            shifting*Mathf.Cos(rot*Mathf.PI/180f) - size * 0.5f, size, size),
                        rotation = rot
                    };
                }
    }
}
