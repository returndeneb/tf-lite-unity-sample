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
        private HandDetect handDetect;
        private FaceMesh faceMesh;
        private HandMesh handMesh;
        private FaceDetect.Result faceDetectResult;
        private FaceMesh.Result faceMeshResult;
        private List<HandDetect.Result> palmResults;
        private HandMesh.Result handMeshResult;
        private PrimitiveDraw draw;
        private readonly Vector3[] imgSize = new Vector3[4];

        private void Start()
        {
            faceDetect = new FaceDetect("mediapipe/face_detection_back.tflite");
            faceMesh = new FaceMesh("mediapipe/face_landmark.tflite");
            handDetect = new HandDetect("mediapipe/palm_detection_builtin_256_float16_quant.tflite");
            handMesh = new HandMesh("mediapipe/hand_landmark.tflite");
            draw = new PrimitiveDraw(Camera.main, gameObject.layer);
            GetComponent<WebCamInput>().onTextureUpdate.AddListener(OnTextureUpdate);
            image.material = faceDetect.TransformMat;
        }
        private void OnDestroy()
        {
            GetComponent<WebCamInput>().onTextureUpdate.RemoveListener(OnTextureUpdate);
            faceDetect?.Dispose();
            faceMesh?.Dispose();
            handDetect?.Dispose();
            handMesh?.Dispose();
            draw?.Dispose();
        }
        private void OnTextureUpdate(Texture texture)
        {
            image.texture = texture;
            DetectFace(texture);
            DetectHand(texture);
        }
        private void Update()
        {
            image.rectTransform.GetWorldCorners(imgSize);
            DrawFace();
            DrawHand();
            draw.Apply();
        }
        private void DetectFace(Texture texture)
        {
            if (texture == null) return;
            faceDetect.Invoke(texture);
            faceDetectResult = faceDetect.GetResults().FirstOrDefault();
            if (faceDetectResult == null) return;
            faceMesh.Invoke(texture, faceDetectResult);
            faceMeshResult = faceMesh.GetResult();
        }
        private void DetectHand(Texture texture)
        {
            if (texture == null) return;
            handDetect.Invoke(texture);
            palmResults = handDetect.GetResults();
            if (palmResults.Count <= 0) return;
            handMesh.Invoke(texture, palmResults[0]);
            handMeshResult = handMesh.GetResult();
        }
        private void DrawFace()
        {
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
            if (handMeshResult == null) return;
            for (var i = 0; i < HandMesh.JOINT_COUNT; i++)
            {
                var p1 = MathTF.Lerp(imgSize[0], imgSize[2], handMeshResult.joints[i]);
                p1.z += handMeshResult.joints[i].z* (imgSize[2].x - imgSize[0].x);
                draw.Point(p1,0.1f);
            }
        }
        
    }
}
