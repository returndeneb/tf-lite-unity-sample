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
        [SerializeField]
        private RawImage image;
        private FaceDetect faceDetect;
        private FaceDetect.Result faceDetectResult;
        private FaceMesh faceMesh;
        private FaceMesh.Result faceMeshResult;
        private PalmDetect palmDetect;
        private List<PalmDetect.Result> palmResults;
        private HandLandmarkDetect landmarkDetect;
        private HandLandmarkDetect.Result landmarkResult;
        private PrimitiveDraw draw;
        private readonly Vector3[] imgSize = new Vector3[4];
        private bool isTextureNull;

        private void Start()
        {
            faceDetect = new FaceDetect("mediapipe/face_detection_back.tflite");
            faceMesh = new FaceMesh("mediapipe/face_landmark.tflite");
            palmDetect = new PalmDetect("mediapipe/palm_detection_builtin_256_float16_quant.tflite");
            landmarkDetect = new HandLandmarkDetect("mediapipe/hand_landmark.tflite");
            
            draw = new PrimitiveDraw(Camera.main, gameObject.layer);
            
            GetComponent<WebCamInput>().onTextureUpdate.AddListener(OnTextureUpdate);
            image.material = faceDetect.TransformMat;
        }
        private void OnDestroy()
        {
            GetComponent<WebCamInput>().onTextureUpdate.RemoveListener(OnTextureUpdate);
            faceDetect?.Dispose();
            faceMesh?.Dispose();
            palmDetect?.Dispose();
            landmarkDetect?.Dispose();
            draw?.Dispose();
        }
        private void Update()
        {
            DrawFace();
            DrawHand();
        }

        private void DrawFace()
        {
            if (image.texture == null) return;
            faceDetect.Invoke(image.texture);
            faceDetectResult = faceDetect.GetResults().FirstOrDefault();
            if (faceDetectResult == null) return;
            faceMesh.Invoke(image.texture, faceDetectResult);
            faceMeshResult = faceMesh.GetResult();
            if (faceMeshResult == null) return;
            
            for (var i = 0; i < faceMeshResult.keyPoints.Length; i++)
            {
                var p = MathTF.Lerp(imgSize[0], imgSize[2], faceMeshResult.keyPoints[i]);
                p.z = faceMeshResult.keyPoints[i].z * (imgSize[2].x - imgSize[0].x) / 2;
                draw.Point(p);
            }
            draw.Apply();
        }
        

        private void DrawHand()
        {
            if (image.texture == null) return;
            palmResults = palmDetect.GetResults();
            
        }
        
        private void OnTextureUpdate(Texture texture)
        {
            image.texture = texture;
            image.rectTransform.GetWorldCorners(imgSize); //Image.rectTranform 데이터가 input, imgSize가 output
        }
        

        private void Draw()
        {
            
            
        }
    }
}
