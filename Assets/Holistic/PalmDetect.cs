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

    public class PalmDetect : BaseImagePredictor<float>
    {
        public struct Result
        {
            public float score;
            public Rect rect;
            public Vector2[] keypoints;
        }

        private const int MaxPalmNum = 4;

        // classificators / scores
        private readonly float[] output0 = new float[2944];

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 17 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7
        private readonly float[,] output1 = new float[2944, 18];
        private readonly List<Result> results = new List<Result>();
        private readonly SsdAnchor[] anchors;

        public PalmDetect(string modelPath) : base(modelPath, Accelerator.NONE)
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
                strides = new int[] { 8, 16, 32, 32, 32 },

                aspectRatios = new float[] { 1.0f },

                reduceBoxesInLowestLayer = false,
                interpolatedScaleAspectRatio = 1.0f,
                fixedAnchorSize = true,
            };

            anchors = SsdAnchorsCalculator.Generate(options);
            Debug.AssertFormat(anchors.Length == 2944, "Anchors count must be 2944");
        }

        public override void Invoke(Texture inputTex)
        {
            // const float OFFSET = 128f;
            // const float SCALE = 1f / 128f;
            // ToTensor(inputTex, input0, OFFSET, SCALE);
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

        public List<Result> GetResults(float scoreThreshold = 0.7f, float iouThreshold = 0.3f)
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

                var sx = output1[i, 0];
                var sy = output1[i, 1];
                var w = output1[i, 2];
                var h = output1[i, 3];

                var cx = sx + anchor.x * width;
                var cy = sy + anchor.y * height;

                cx /= width;
                cy /= height;
                w /= width;
                h /= height;

                var keyPoints = new Vector2[7];
                for (var j = 0; j < 7; j++)
                {
                    var lx = output1[i, 4 + (2 * j) + 0];
                    var ly = output1[i, 4 + (2 * j) + 1];
                    lx += anchor.x * width;
                    ly += anchor.y * height;
                    lx /= width;
                    ly /= height;
                    keyPoints[j] = new Vector2(lx, ly);
                }

                results.Add(new Result()
                {
                    score = score,
                    rect = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h),
                    keypoints = keyPoints,
                });

            }

            return NonMaxSuppression(results, iouThreshold);
        }

        private static List<Result> NonMaxSuppression(IEnumerable<Result> palms, float iouThreshold)
        {
            var filtered = new List<Result>();

            foreach (var originalPalm in palms.OrderByDescending(o => o.score))
            {
                var ignoreCandidate = false;
                foreach (var newPalm in filtered)
                {
                    var iou = originalPalm.rect.IntersectionOverUnion(newPalm.rect);
                    if (!(iou >= iouThreshold)) continue;
                    ignoreCandidate = true;
                    break;
                }

                if (ignoreCandidate) continue;
                filtered.Add(originalPalm);
                if (filtered.Count >= MaxPalmNum)
                {
                    break;
                }
            }

            return filtered;
        }

    }
}
