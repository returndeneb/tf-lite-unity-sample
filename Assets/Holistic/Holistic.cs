using System.Collections.Generic;
using System.Linq;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;
using System.Collections;

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
        private readonly Vector3[] worldJoints = new Vector3[HandLandmarkDetect.JOINT_COUNT];
        private bool isTextureNull;

        private void Start()
        {
            faceDetect = new FaceDetect("mediapipe/face_detection_back.tflite");
            faceMesh = new FaceMesh("mediapipe/face_landmark.tflite");
            palmDetect = new PalmDetect("mediapipe/palm_detection_builtin_256_float16_quant.tflite");
            landmarkDetect = new HandLandmarkDetect("mediapipe/hand_landmark.tflite");
            
            draw = new PrimitiveDraw(Camera.main, gameObject.layer);
            
            GetComponent<WebCamInput>().onTextureUpdate.AddListener(OnTextureUpdate);
            
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
            // DrawFace();
            DrawHand();
            draw.Apply();
            
        }

        private void DrawFace()
        {
            
            if (image.texture == null) return;
            image.material = faceDetect.TransformMat;
            image.rectTransform.GetWorldCorners(imgSize);
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
        }
        
        private void DrawHand()
        {
            if (image.texture == null) return;
            image.material = palmDetect.TransformMat;
            image.rectTransform.GetWorldCorners(imgSize);
            palmDetect.Invoke(image.texture);
            // palmResults = palmDetect.GetResults();
            
            // if (palmResults.Count <= 0) return;
            // landmarkDetect.Invoke(image.texture, palmResults[0]);
            // landmarkResult = landmarkDetect.GetResult();
            // if (landmarkResult == null) return;
            
            // for (var i = 0; i < HandLandmarkDetect.JOINT_COUNT; i++)
            // {
            //     var p1 = MathTF.Lerp(imgSize[0], imgSize[2], landmarkResult.joints[i]);
            //     p1.z += (landmarkResult.joints[i].z - 0.5f) * (imgSize[2].x - imgSize[0].x);
            //     draw.Point(p1,0.1f);
            // }
        }
        
        private void OnTextureUpdate(Texture texture)
        {
        //     UnityMainThreadDispatcher.Instance().Enqueue(() =>
        //     {
        //         image.texture = texture;
        //         print("test");
        //     });
        }
        
    }
}
