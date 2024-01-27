using System.Collections.Generic;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

namespace Samples.FaceMesh
{
    public class FaceDetectionSample : MonoBehaviour
    {

        [SerializeField]
        private RawImage cameraView;

        private FaceDetect faceDetect;
        private List<FaceDetect.Result> results;
        private PrimitiveDraw draw;
        private readonly Vector3[] rtCorners = new Vector3[4];

        private void Start()
        {
            print("testtssetss");
            faceDetect = new FaceDetect("mediapipe/face_detection_back.tflite");
            draw = new PrimitiveDraw( gameObject.layer);
            
            GetComponent<WebCamInput>().onTextureUpdate.AddListener(OnTextureUpdate);
            
            cameraView.material = faceDetect.TransformMat;
            cameraView.rectTransform.GetWorldCorners(rtCorners);
        }

        private void OnDestroy()
        {
            GetComponent<WebCamInput>().onTextureUpdate.RemoveListener(OnTextureUpdate);
            faceDetect?.Dispose();
            draw?.Dispose();
        }

        private void Update()
        {
            DrawResults(results);
        }
        
        private void OnTextureUpdate(Texture texture)
        {
            cameraView.texture = texture;
            faceDetect.Invoke(texture);
            results = faceDetect.GetResults();
        }
        

        private void DrawResults(List<FaceDetect.Result> faceResults)
        {
            if (faceResults == null || faceResults.Count == 0) return;
            foreach (var result in faceResults)
            {
                var rect = MathTF.Lerp(rtCorners[0], rtCorners[2], result.rect, true);
                draw.Rect(rect, 0.05f);
                foreach (var p in result.keypoints)
                {
                    draw.Point(MathTF.Lerp(rtCorners[0], rtCorners[2], new Vector3(p.x, 1f - p.y, 0)), 0.1f);
                }
            }
            draw.Apply();
        }
    }
}
