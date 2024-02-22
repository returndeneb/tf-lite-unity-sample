using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using UnityEngine;

namespace Holistic
{
    public sealed class FaceDetect : ImageInterpreter<float>
    {
        public class Result
        {
            public float score;
            public Rect rect;
            public float rotation;
        }

        private const int KeyPointsNum = 6;

        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 15 are 6 keypoint x and y coordinates: x0,y0,x1,y1,x2,y2,x3,y3
        private readonly float[,] output0 = new float[896, 16];

        // scores
        private readonly float[] output1 = new float[896];

        private readonly SsdAnchor[] anchors;
        private readonly List<Result> results = new ();
        private readonly List<Result> filteredResults = new ();

        public FaceDetect(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            var options = new SsdAnchorsCalculator.Options
            {
                inputSizeWidth = 128,
                inputSizeHeight = 128,

                minScale = 0.1484375f,
                maxScale = 0.75f,

                anchorOffsetX = 0.5f,
                anchorOffsetY = 0.5f,

                numLayers = 4,
                featureMapWidth = Array.Empty<int>(),
                featureMapHeight = Array.Empty<int>(),
                strides = new[] { 8, 16, 16, 16 },

                aspectRatios = new[] { 1.0f },

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

        public IEnumerable<Result> GetResults(float scoreThreshold = 0.7f)
        {
            results.Clear();

            for (var a = 0; a < anchors.Length; a++)
            {
                var score = MathTF.Sigmoid(output1[a]);
                if (score < scoreThreshold)
                {
                    continue;
                }
                var anchor = anchors[a];
                var x = output0[a, 0]/width + anchor.x;
                var y = 1-(output0[a, 1]/height + anchor.y);
                var w = output0[a, 2]/width*1.6f;
                var h = output0[a, 3]/height*1.6f;
                
                var keyPoints = new Vector2[2];
                for (var i = 0; i < 2; i++)
                {
                    var xi = output0[a, 4 + (2 * i) + 0]/width+anchor.x;
                    var yi = output0[a, 4 + (2 * i) + 1]/height+anchor.y;
                    keyPoints[i] = new Vector2(xi, yi);
                }
                var vec = keyPoints[0] - keyPoints[1];
                results.Add(new Result()
                {
                    score = score,
                    rect = new Rect(x - w * 0.5f, y - h * 0.5f, w, h),
                    rotation =  Mathf.Atan2(vec.y, vec.x)*Mathf.Rad2Deg
                });
            }
            return NonMaxSuppression();
        }

        private IEnumerable<Result> NonMaxSuppression(float iouThreshold=0.3f)
        {
            filteredResults.Clear();
            foreach (var result in results.OrderByDescending(o => o.score))
            {
                var ignoreCandidate = filteredResults.Select(newResult => 
                    result.rect.IntersectionOverUnion(newResult.rect)).Any(iou => iou >= iouThreshold);
                if (ignoreCandidate) continue;
                filteredResults.Add(result);
                if (filteredResults.Count >= 100) break;
            }
            return filteredResults;
        }
    }
}
