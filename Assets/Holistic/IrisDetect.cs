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
        private Vector2 FaceScale { get; set; } = new(1.6f, 1.6f);
        private readonly Result result;
        private const int KeypointCount = 468;
        private readonly float[,] output0 = new float[KeypointCount, 3]; // key points
        private readonly float[] output1 = new float[1]; // score
        public IrisMesh(string modelPath, InterpreterOptions options) : base(modelPath, options)
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
        
    }
}