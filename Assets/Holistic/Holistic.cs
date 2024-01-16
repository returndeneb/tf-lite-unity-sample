using System.Collections.Generic;
using System.Linq;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;

namespace Holistic
{
    public class Holistic : MonoBehaviour
    {
        private RawImage image;
        private FaceDetect faceDetect;
        private FaceDetect.Result faceDetectResult;
        private FaceMesh faceMesh;
        private FaceMesh.Result faceMeshResult;
        
        private PrimitiveDraw draw;
        private readonly Vector3[] imgSize = new Vector3[4];

        private void Start()
        {
            faceDetect = new FaceDetect("mediapipe/face_detection_back.tflite");
            faceMesh = new FaceMesh("mediapipe/face_landmark.tflite");
            draw = new PrimitiveDraw(Camera.main, gameObject.layer);
            
            GetComponent<WebCamInput>().onTextureUpdate.AddListener(OnTextureUpdate);
            image.material = faceDetect.TransformMat;
        }
        private void OnDestroy()
        {
            GetComponent<WebCamInput>().onTextureUpdate.RemoveListener(OnTextureUpdate);
            faceDetect?.Dispose();
            faceMesh?.Dispose();
            draw?.Dispose();
        }
        private void Update()
        {
            faceDetect.Invoke(image.texture);
            faceDetectResult = faceDetect.GetResults().FirstOrDefault();
            if (faceDetectResult == null) return;
            faceMesh.Invoke(image.texture, faceDetectResult);
            // faceMeshResult = faceMesh.GetResult();
            // if (faceMeshResult == null) return;
            Draw();
        }
        
        private void OnTextureUpdate(Texture texture)
        {
            // print(texture.height);
            image.texture = texture;
            image.rectTransform.GetWorldCorners(imgSize); //Image.rectTranform 데이터가 input, imgSize가 output
        }
        

        private void Draw()
        {
            var rect = MathTF.Lerp(imgSize[0], imgSize[2], faceDetectResult.rect, true);
            draw.Rect(rect);
            foreach (var p in faceDetectResult.keyPoints)
            {
                var keyPoint = MathTF.Lerp(imgSize[0], imgSize[2], new Vector2(p.x, 1f - p.y));
                draw.Point(keyPoint);
            }
            draw.Apply();
        }
    }
}
