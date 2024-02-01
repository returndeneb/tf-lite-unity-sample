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
        
        private Matrix4x4 cropMatrix;
        private Vector2 EyeScale { get; set; } = new(1.1f, 1.1f);
        private readonly Result result;
        private const int KeypointCount = 5;
        private readonly float[,] output = new float[1,15]; // 동공 key points 왼쪽
        private Rect cropRect;
        public IrisMesh(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            result = new Result()
            {
                keyPoints = new Vector3[KeypointCount],
                score = 0,
            };
        }

        public void Invoke(Texture inputTex, FaceMesh.Result face,bool left)
        {
            var eye = left? new List<Vector2>{face.keyPoints[33], face.keyPoints[133]}: new List<Vector2>{face.keyPoints[362], face.keyPoints[263]};
            var size = Mathf.Max(eye[0].x - eye[1].x, eye[0].y - eye[1].y);
            var center = (eye[0] + eye[1]) / 2f;
            cropRect = new Rect(center.x - size / 2f, center.y - size / 2f, size, size);
                
            cropMatrix = RectTransformationCalculator.CalcMatrix(new RectTransformationCalculator.Options
            {
                rect = cropRect,
                rotationDegree = 180f,
                scale = EyeScale,
            });

            var rt = resizer.Resize(
                inputTex, resizeOptions.width, resizeOptions.height,
                cropMatrix,
                TextureResizer.GetTextureSt(inputTex, resizeOptions));
            
            ToTensor(rt, inputTensor, false);
            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(1, output);
        }
        
        public Result GetResult()
        {
            const float scale = 1/64f ;
            var mtx = cropMatrix.inverse;

            for (var i = 0; i < 5; i++)
            {
                result.keyPoints[i] = mtx.MultiplyPoint3x4(new Vector3(
                    output[0,i*3] * scale,
                    1-output[0,i*3+1] * scale,
                    output[0,i*3+2] * scale
                ));
            }
            return result;
        }
        
    }
}