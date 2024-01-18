using System;
using System.Collections.Generic;
using System.Linq;
using TensorFlowLite;
using UnityEngine;

namespace Holistic
{
    public class FaceDetect : BaseImagePredictor<float>
    {
        private enum KeyPoint
        {
            RightEye,  //  0
            LeftEye, //  1
            Nose, //  2
            Mouth, //  3
            RightEar, //  4
            LeftEar, //  5
        }

        public class Result
        {
            public float score;
            public Rect rect;
            public Vector2[] keyPoints;

            public Vector2 RightEye => keyPoints[(int)KeyPoint.RightEye];
            public Vector2 LeftEye => keyPoints[(int)KeyPoint.LeftEye];
        }

        private const int KeyPointSize = 6;

        private const int MaxFaceNum = 100;

        // regress / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 15 are 6 keypoint x and y coordinates: x0,y0,x1,y1,x2,y2,x3,y3
        private readonly float[,] output0 = new float[896, 16];

        // classifications / scores
        private readonly float[] output1 = new float[896];

        private readonly SsdAnchor[] anchors;
        private readonly List<Result> results = new ();
        private readonly List<Result> filteredResults = new ();

        public FaceDetect(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            var options = new SsdAnchorsCalculator.Options()
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
                strides = new int[] { 8, 16, 16, 16 },

                aspectRatios = new float[] { 1.0f },

                reduceBoxesInLowestLayer = false,
                interpolatedScaleAspectRatio = 1.0f,
                fixedAnchorSize = true,
            };

            anchors = SsdAnchorsCalculator.Generate(options);
            UnityEngine.Debug.AssertFormat(anchors.Length == 896, $"Anchors count must be 896, but was {anchors.Length}");
        }

        public override void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputTensor);

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public List<Result> GetResults(float scoreThreshold = 0.7f, float iouThreshold = 0.3f)
        {
            results.Clear();

            for (var i = 0; i < anchors.Length; i++)
            {
                var score = MathTF.Sigmoid(output1[i]);
                if (score < scoreThreshold)
                {
                    continue;
                }
                var anchor = anchors[i];

                var sx = output0[i, 0];
                var sy = output0[i, 1];
                var w = output0[i, 2];
                var h = output0[i, 3];

                var cx = sx + anchor.x * width;
                var cy = sy + anchor.y * height;

                cx /= (float)width;
                cy /= (float)height;
                w /= (float)width;
                h /= (float)height;

                var keyPoints = new Vector2[KeyPointSize];
                for (var j = 0; j < KeyPointSize; j++)
                {
                    var lx = output0[i, 4 + (2 * j) + 0];
                    var ly = output0[i, 4 + (2 * j) + 1];
                    lx += anchor.x * width;
                    ly += anchor.y * height;
                    lx /= (float)width;
                    ly /= (float)height;
                    keyPoints[j] = new Vector2(lx, ly);
                }
                results.Add(new Result()
                {
                    score = score,
                    rect = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h),
                    keyPoints = keyPoints,
                });
            }
            return NonMaxSuppression(iouThreshold);
        }

        private List<Result> NonMaxSuppression(float iouThreshold)
        {
            filteredResults.Clear();
            // FIXME LinQ allocs GC each frame
            // Use sorted list
            foreach (var original in results.OrderByDescending(o => o.score))
            {
                var ignoreCandidate = filteredResults.Select(newResult => original.rect.IntersectionOverUnion(newResult.rect)).Any(iou => iou >= iouThreshold);

                if (ignoreCandidate) continue;
                filteredResults.Add(original);
                if (filteredResults.Count >= MaxFaceNum)
                {
                    break;
                }
            }
            return filteredResults;
        }
    }
}
