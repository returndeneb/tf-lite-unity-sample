using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.Assertions;

namespace Holistic
{
    public sealed class PoseDetect : ImageInterpreter<float>
    {
        public class Result : System.IComparable<Result>
        {
            public float score;
            public Rect rect;
            public Vector2[] keyPoints;

            public static Result Negative => new Result() { score = -1, };

            public int CompareTo(Result other)
            {
                return score > other.score ? -1 : 1;
            }
        }

        private const int MaxPoseNum = 100;
        private const int AnchorLength = 2254;
        private readonly int keyPointsCount;

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 11 are 4 keypoints x and y coordinates: x0,y0,x1,y1,x2,y2,x3,y3
        private readonly float[,] output0 = new float[AnchorLength, 12];

        // classificators / scores
        private readonly float[] output1 = new float[AnchorLength];

        private readonly SsdAnchor[] anchors;
        private readonly SortedSet<Result> results = new SortedSet<Result>();

        public PoseDetect(string modelPath) : base(modelPath, Accelerator.NONE)
        {
            resizeOptions.aspectMode = AspectMode.Fit;

            var anchorOptions = new SsdAnchorsCalculator.Options()
            {
                inputSizeWidth = width,
                inputSizeHeight = height,

                minScale = 0.1484375f,
                maxScale = 0.75f,

                anchorOffsetX = 0.5f,
                anchorOffsetY = 0.5f,

                numLayers = 5,
                featureMapWidth = new int[0],
                featureMapHeight = new int[0],
                strides = new int[] { 8, 16, 32, 32, 32 },

                aspectRatios = new float[] { 1.0f },

                reduceBoxesInLowestLayer = false,
                interpolatedScaleAspectRatio = 1.0f,
                fixedAnchorSize = true,
            };

            anchors = SsdAnchorsCalculator.Generate(anchorOptions);
            Assert.AreEqual(anchors.Length, AnchorLength,
                $"Anchors count must be {AnchorLength}, but was {anchors.Length}");

            // Get Keypoint Mode
            var odim0 = interpreter.GetOutputTensorInfo(0).shape;
            keyPointsCount = (odim0[2] - 4) / 2;
        }

        public void Invoke(Texture inputTex)
        {
            ToTensor(inputTex, inputTensor);

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public async UniTask<Result> InvokeAsync(Texture inputTex, CancellationToken cancellationToken, PlayerLoopTiming timing = PlayerLoopTiming.Update)
        {
            await ToTensorAsync(inputTex, inputTensor, cancellationToken);
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);

            var results = GetResults();

            await UniTask.SwitchToMainThread(timing, cancellationToken);
            return results;
        }

        public Result GetResults()
        {
            results.Clear();

            for (var i = 0; i < anchors.Length; i++)
            {
                var score = MathTF.Sigmoid(output1[i]);
                if (score < 0.5f)
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

                cx /= width;
                cy /= height;
                w /= width;
                h /= height;

                var keypoints = new Vector2[keyPointsCount];
                for (var j = 0; j < keyPointsCount; j++)
                {
                    var lx = output0[i, 4 + (2 * j) + 0];
                    var ly = output0[i, 4 + (2 * j) + 1];
                    lx += anchor.x * width;
                    ly += anchor.y * height;
                    lx /= width;
                    ly /= height;
                    keypoints[j] = new Vector2(lx, ly);
                }

                results.Add(new Result()
                {
                    score = score,
                    rect = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h),
                    keyPoints = keypoints,
                });
            }

            // No result
            return results.Count == 0 ? Result.Negative : NonMaxSuppression(results, 0.5f).First();
        }

        private static readonly List<Result> nonMaxSupressionCache = new List<Result>();
        private static List<Result> NonMaxSuppression(SortedSet<Result> results, float iouThreshold)
        {
            nonMaxSupressionCache.Clear();
            foreach (Result original in results)
            {
                bool ignoreCandidate = false;
                foreach (Result newResult in nonMaxSupressionCache)
                {
                    float iou = original.rect.IntersectionOverUnion(newResult.rect);
                    if (iou >= iouThreshold)
                    {
                        ignoreCandidate = true;
                        break;
                    }
                }

                if (!ignoreCandidate)
                {
                    nonMaxSupressionCache.Add(original);
                    if (nonMaxSupressionCache.Count >= MaxPoseNum)
                    {
                        break;
                    }
                }
            }

            return nonMaxSupressionCache;
        }
    }
}
