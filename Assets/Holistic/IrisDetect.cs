using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using TensorFlowLite;


namespace Holistic
{
    public class IrisMesh: ImageInterpreter<float>
    {
        public class Result
        {
            public Vector3[] keyPoints;
            public float score;
            
        }
        
        private Matrix4x4 cropMatrix_left;
        private Matrix4x4 cropMatrix_right;
        private Vector2 EyeScale { get; set; } = new(1.1f, 1.1f);
        private readonly Result result;
        private const int KeypointCount = 5;
        private readonly float[,] output0 = new float[1,213]; // ???
        private readonly float[,] output1 = new float[1,15]; // key points
        public IrisMesh(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            result = new Result()
            {
                keyPoints = new Vector3[KeypointCount],
                score = 0,
            };
        }

        public void Invoke(Texture inputTex, FaceMesh.Result face)
        {
            var leftEye = new List<Vector2>{face.keyPoints[33], face.keyPoints[133]};
            var leftSize = Mathf.Max(leftEye[0].x - leftEye[1].x, leftEye[0].y - leftEye[1].y);
            var leftCenter = (leftEye[0] + leftEye[1]) / 2f;
            var leftRect = new Rect(leftCenter.x - leftSize / 2f, leftCenter.y - leftSize / 2f, leftSize, leftSize);
            var rightEye = new List<Vector3>{face.keyPoints[362], face.keyPoints[264]};
           
            cropMatrix_left = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options
            {
                rect = leftRect,
                rotationDegree = 0f,
                // shift = FaceShift,
                scale = EyeScale,
            });
        
        
            var rt = resizer.Resize(
                inputTex, resizeOptions.width, resizeOptions.height,
                cropMatrix_left,
                TextureResizer.GetTextureSt(inputTex, resizeOptions));
            
            ToTensor(rt, inputTensor, false);

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }
        
        public Result GetResult()
        {
            const float scale = 1/64f ;
            var mtx = cropMatrix_left.inverse;

            result.score = output1[0,0];
            for (var i = 0; i < KeypointCount; i++)
            {
                result.keyPoints[i] = mtx.MultiplyPoint3x4(new Vector3(
                    output0[0,i*3] * scale,
                    1-output0[0,i*3+1] * scale,
                    output0[0,i*3+2] * scale
                ));
            }
            return result;
        }
        
    }
}