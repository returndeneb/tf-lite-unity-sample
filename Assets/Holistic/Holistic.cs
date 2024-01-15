using System.Collections.Generic;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

namespace Holistic
{
    public class Holistic : MonoBehaviour
    {

        [SerializeField]
        private RawImage cameraView;
        private FaceDetect faceDetect;
        private FaceMesh faceMesh;
        
        private PrimitiveDraw draw;
        private readonly Vector3[] canvasSize = new Vector3[4];

        private void Start()
        {
            faceDetect = new FaceDetect("mediapipe/face_detection_back.tflite");
            faceMesh = new FaceMesh("mediapipe/face_landmark.tflite");
            draw = new PrimitiveDraw(Camera.main, gameObject.layer);
            
            GetComponent<WebCamInput>().onTextureUpdate.AddListener(OnTextureUpdate);
            cameraView.material = faceDetect.TransformMat;
        }
        private void OnDestroy()
        {
            GetComponent<WebCamInput>().onTextureUpdate.RemoveListener(OnTextureUpdate);
            faceDetect?.Dispose();
            draw?.Dispose();
        }
        private void Update()
        {
            DrawFaceDetect(faceDetect.GetResults());
        }
        
        private void OnTextureUpdate(Texture texture)
        {
            faceDetect.Invoke(texture);
            cameraView.texture = texture;
            cameraView.rectTransform.GetWorldCorners(canvasSize);
        }
        

        private void DrawFaceDetect(List<FaceDetect.Result> faceDetectResults)
        {
            if (faceDetectResults == null || faceDetectResults.Count == 0) return;
            foreach (var result in faceDetectResults)
            {
                var rect = MathTF.Lerp(canvasSize[0], canvasSize[2], result.rect, true);
                draw.Rect(rect, 0.05f);
                foreach (var p in result.keyPoints)
                {
                    draw.Point(MathTF.Lerp(canvasSize[0], canvasSize[2], new Vector3(p.x, 1f - p.y, 0)), 0.1f);
                }
            }
            draw.Apply();
        }
    }
}
