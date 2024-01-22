using System.Collections.Generic;
using System.Linq;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

namespace Holistic
{
    public class Main : MonoBehaviour
    {
        [SerializeField]
        private RawImage image;
        
        private FaceDetect faceDetect;
        private HandDetect handDetect;
        private PoseDetect poseDetect;
        
        private FaceMesh faceMesh;
        private HandMesh handMesh;
        private PoseMesh poseMesh;
        
        private FaceDetect.Result faceDetectResult;
        private List<HandDetect.Result> handDetectResults;
        private PoseDetect.Result poseDetectResult;
        
        private FaceMesh.Result faceMeshResult;
        private HandMesh.Result handMeshResult;
        private PoseMesh.Result poseMeshResult;
        
        private PrimitiveDraw draw;
        private readonly Vector3[] imgSize = new Vector3[4];
        private Vector4[] viewportLandmarks;
        private void Start()
        {
            faceDetect = new FaceDetect("mediapipe/face_detection_back.tflite");
            handDetect = new HandDetect("mediapipe/palm_detection_builtin_256_float16_quant.tflite");
            poseDetect = new PoseDetect("mediapipe/pose_detection.tflite");
            
            faceMesh = new FaceMesh("mediapipe/face_landmark.tflite");
            handMesh = new HandMesh("mediapipe/hand_landmark.tflite");
            poseMesh = new PoseMesh("mediapipe/pose_landmark_lite.tflite");
            
            draw = new PrimitiveDraw(Camera.main, gameObject.layer);
            viewportLandmarks = new Vector4[PoseMesh.LandmarkCount];
            GetComponent<WebCamInput>().onTextureUpdate.AddListener(OnTextureUpdate);
            
            image.material = faceDetect.TransformMat;
        }
        private void OnDestroy()
        {
            GetComponent<WebCamInput>().onTextureUpdate.RemoveListener(OnTextureUpdate);
            faceDetect?.Dispose();
            handDetect?.Dispose();
            poseDetect?.Dispose();
            
            faceMesh?.Dispose();
            handMesh?.Dispose();
            poseMesh?.Dispose();
            
            draw?.Dispose();
        }
        private void OnTextureUpdate(Texture texture)
        {
            image.texture = texture;
            DetectFace(texture);
            DetectHand(texture);
            DetectPose(texture);
        }
        private void Update()
        {
            image.rectTransform.GetWorldCorners(imgSize);
            DrawFace();
            DrawHand();
            DrawPose();
            draw.Apply();
        }
        private void DetectFace(Texture texture)
        {
            if (texture == null) return;
            if (faceDetectResult == null)
            {
                faceDetect.Invoke(texture);
                faceDetectResult = faceDetect.GetResults().FirstOrDefault();
                if (faceDetectResult == null) return;
            }
            faceMesh.Invoke(texture, faceDetectResult);
            faceMeshResult = faceMesh.GetResult();
            faceDetectResult = FaceMesh.LandmarkToDetection(faceMeshResult);
        }
        private void DetectHand(Texture texture)
        {
            if (texture == null) return;
            handDetect.Invoke(texture);
            handDetectResults = handDetect.GetResults();
            if (handDetectResults.Count <= 0) return;
            handMesh.Invoke(texture, handDetectResults[0]);
            handMeshResult = handMesh.GetResult();
        }

        private void DetectPose(Texture texture)
        {
            if (texture == null) return;
            if (poseDetectResult?.keypoints == null)
            {
                poseDetect.Invoke(texture);
                poseDetectResult = poseDetect.GetResults();
                if (poseDetectResult?.keypoints == null) return;
            }
            poseMeshResult = poseMesh.Invoke(texture, poseDetectResult);
            poseDetectResult = PoseMesh.LandmarkToDetection(poseMeshResult);
        }
            
        private void DrawFace()
        {
            if (faceMeshResult == null) return;
            for (var i = 0; i < faceMeshResult.keyPoints.Length; i++)
            {
                var kp = faceMeshResult.keyPoints[i];
                kp.y = 1 - kp.y;
                var p = MathTF.Lerp(imgSize[0], imgSize[2], kp);
                p.z = faceMeshResult.keyPoints[i].z * (imgSize[2].x - imgSize[0].x) / 2;
                draw.Point(p);
            }
        }
        private void DrawHand()
        {
            if (handMeshResult == null) return;
            for (var i = 0; i < HandMesh.JointCount; i++)
            {
                var p1 = MathTF.Lerp(imgSize[0], imgSize[2], handMeshResult.joints[i]);
                p1.z += handMeshResult.joints[i].z* (imgSize[2].x - imgSize[0].x);
                draw.Point(p1,0.1f);
            }
        }
        private void DrawPose(float visibilityThreshold=0.5f)
        {
            if (poseMeshResult == null) return;
            var landmarks = poseMeshResult.viewportLandmarks;
            for (var i = 0; i < landmarks.Length; i++)
            {
                var p = Camera.main.ViewportToWorldPoint(landmarks[i]);
                viewportLandmarks[i] = new Vector4(p.x, p.y, p.z, landmarks[i].w);
                if (viewportLandmarks[i].w > visibilityThreshold) 
                    draw.Cube(viewportLandmarks[i], 0.2f);
            }
            
            var connections = PoseMesh.Connections;
            for (var i = 0; i < connections.Length; i += 2)
            {
                var a = viewportLandmarks[connections[i]];
                var b = viewportLandmarks[connections[i + 1]];
                if (a.w > visibilityThreshold || b.w > visibilityThreshold)
                {
                    draw.Line3D(a, b, 0.05f);
                }
            }
            draw.Apply();
        }
        
    }
}
