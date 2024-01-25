using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.Rendering;

namespace Holistic
{
    public sealed class HandDetect : ImageInterpreter<float>
    {
        public struct Result
        {
            public float score;
            public Rect rect;
            public float rotation;
        }

        private const int MaxPalmNum = 2;

        //scores
        private readonly float[] output0 = new float[2944];

        // key points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 17 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7
        private readonly float[,] output1 = new float[2944, 18];
        
        private readonly SsdAnchor[] anchors;
        private readonly List<Result> results = new();
        
        public HandDetect(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            var options = new SsdAnchorsCalculator.Options()
            {
                inputSizeWidth = 256,
                inputSizeHeight = 256,

                minScale = 0.1171875f,
                maxScale = 0.75f,

                anchorOffsetX = 0.5f,
                anchorOffsetY = 0.5f,

                numLayers = 5,
                featureMapWidth = Array.Empty<int>(),
                featureMapHeight = Array.Empty<int>(),
                strides = new [] { 8, 16, 32, 32, 32 },

                aspectRatios = new [] { 1.0f },

                reduceBoxesInLowestLayer = false,
                interpolatedScaleAspectRatio = 1.0f,
                fixedAnchorSize = true,
            };
            anchors = SsdAnchorsCalculator.Generate(options);
        }

        public void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputTensor);

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
            
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public async UniTask<List<Result>> InvokeAsync(Texture inputTex, CancellationToken cancellationToken)
        {
            await ToTensorAsync(inputTex, inputTensor, cancellationToken);
            await UniTask.SwitchToThreadPool();
        
            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();
        
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        
            var invokeAsync = GetResults();
        
            await UniTask.SwitchToMainThread(cancellationToken);
            return invokeAsync;
        }

        public List<Result> GetResults(float scoreThreshold = 0.7f)
        {
            results.Clear();

            for (var i = 0; i < anchors.Length; i++)
            {
                var score = MathTF.Sigmoid(output0[i]);
                if (score < scoreThreshold)
                {
                    continue;
                }

                var anchor = anchors[i];

                var cx = output1[i, 0] / width + anchor.x;
                var cy = output1[i, 1] / height + anchor.y;
                var w = output1[i, 2] / width;
                var h = output1[i, 3] / height;
                
                var keyPoints = new Vector2[3];
                for (var j = 0; j < 3; j++)
                {
                    var lx = output1[i, 4 + (2 * j) + 0]/width +anchor.x;
                    var ly = output1[i, 4 + (2 * j) + 1]/height +anchor.y;
                    
                    keyPoints[j] = new Vector2(lx, ly);
                }
                var vec = keyPoints[0] - keyPoints[2];
                var rot = -90f - Mathf.Atan2(vec.y, vec.x) * Mathf.Rad2Deg;
                const float shifting = 0.2f / 2.8f;

                results.Add(new Result()
                {
                    score = score,
                    rect = new Rect(cx + shifting*Mathf.Sin(rot*Mathf.PI/180f)- w * 0.5f, cy + 
                        shifting*Mathf.Cos(rot*Mathf.PI/180f) - h * 0.5f, w, h),

                    rotation =  rot
                });
            }

            return NonMaxSuppression(results);
        }

        private static List<Result> NonMaxSuppression(IEnumerable<Result> palms, float iouThreshold=0.3f)
        {
            var filtered = new List<Result>();

            foreach (var originalPalm in palms.OrderByDescending(o => o.score))
            {
                var ignoreCandidate = filtered.Select(newPalm => 
                    originalPalm.rect.IntersectionOverUnion(newPalm.rect)).Any(iou => iou >= iouThreshold);
                if (ignoreCandidate) continue;
                filtered.Add(originalPalm);
                if (filtered.Count >= MaxPalmNum) break;
            }
            return filtered;
        }

    }
}
