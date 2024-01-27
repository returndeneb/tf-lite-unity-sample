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
        private HandMesh handMesh2;
        private PoseMesh poseMesh;
        
        private FaceDetect.Result faceDetectResult;
        private List<HandDetect.Result> handDetectResults;
        private PoseDetect.Result poseDetectResult;
        
        private FaceMesh.Result faceMeshResult;
        private HandMesh.Result handMeshResult;
        private HandMesh.Result handMeshResult2;
        private PoseMesh.Result poseMeshResult;
        
        private PrimitiveDraw draw;
        private readonly Vector3[] imgSize = new Vector3[4];
        private Vector4[] viewportLandmarks;
        private void Start()
        {
            faceDetect = new FaceDetect("face_detection.tflite");
            handDetect = new HandDetect("palm_detection.tflite");
            poseDetect = new PoseDetect("pose_detection.tflite");
            
            faceMesh = new FaceMesh("face_landmark.tflite");
            handMesh = new HandMesh("hand_landmark.tflite");
            handMesh2 = new HandMesh("hand_landmark.tflite");
            poseMesh = new PoseMesh("pose_landmark_lite.tflite");
            
            draw = new PrimitiveDraw();
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
            handMesh2?.Dispose();
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
            faceDetectResult = faceMeshResult.score < 0f ? null : FaceMesh.LandmarkToDetection(faceMeshResult);
        }
        private void DetectHand(Texture texture)
        {
            if (texture == null) return;
            if (handDetectResults is not { Count: > 0 } )
            {
                
                handDetect.Invoke(texture);
                handDetectResults = handDetect.GetResults();
                if (handDetectResults.Count <= 0) return;
            }
            handMesh.Invoke(texture, handDetectResults[0]);
            handMeshResult = handMesh.GetResult();
            
            
            if (handMeshResult.score < 0.9f)
            {
                handDetectResults = null;
                return;
            }
            handDetectResults[0] = HandMesh.LandmarkToDetection(handMeshResult);
            
            if (handDetectResults.Count <= 1) return;
            handMesh2.Invoke(texture, handDetectResults[1]);
            handMeshResult2 = handMesh2.GetResult();
            if (handMeshResult2.score < 0.9f)
            {
                handDetectResults = null;
                return;
            }
            handDetectResults[1] = HandMesh.LandmarkToDetection(handMeshResult2);
        }

        private void DetectPose(Texture texture)
        {
            if (texture == null) return;
            if (poseDetectResult?.keyPoints == null)
            {
                poseDetect.Invoke(texture);
                poseDetectResult = poseDetect.GetResults();
                if (poseDetectResult?.keyPoints == null) return;
            }
            poseMeshResult = poseMesh.Invoke(texture, poseDetectResult);
            poseDetectResult = poseMeshResult.score < 0.9f?null:PoseMesh.LandmarkToDetection(poseMeshResult);
        }
            
        private void DrawFace()
        {
            if (faceMeshResult == null) return;
            for (var i = 0; i < faceMeshResult.keyPoints.Length; i++)
            {
                var kp = faceMeshResult.keyPoints[i];
                var p = MathTF.Lerp(imgSize[0], imgSize[2], kp,true);
                p.z = faceMeshResult.keyPoints[i].z * (imgSize[2].x - imgSize[0].x) / 2;
                draw.Point(p);
            }
        }
        private void DrawHand()
        {
            if (handMeshResult != null)
            {
                
                for (var i = 0; i < HandMesh.JointCount; i++)
                {
                    var kp = handMeshResult.keyPoints[i];
                    var p1 = MathTF.Lerp(imgSize[0], imgSize[2], kp,true);
                    p1.z += handMeshResult.keyPoints[i].z* (imgSize[2].x - imgSize[0].x);
                    
                    draw.Point(p1,0.1f);
                }
            }
            
            if (handMeshResult2 == null) return;
            
            for (var i = 0; i < HandMesh.JointCount; i++)
            {
                var p1 = MathTF.Lerp(imgSize[0], imgSize[2], handMeshResult2.keyPoints[i],true);
                p1.z += handMeshResult2.keyPoints[i].z* (imgSize[2].x - imgSize[0].x);
                
                draw.Point(p1,0.1f);
            }
        }
        private void DrawPose(float visibilityThreshold=0.5f)
        {
            if (poseMeshResult == null) return;
            var landmarks = poseMeshResult.viewportLandmarks;
            for (var i = 0; i < landmarks.Length; i++)
            {
                // var p = Camera.main.ViewportToWorldPoint(landmarks[i]);
                var p = MathTF.Lerp(imgSize[0],imgSize[2],landmarks[i]);
                viewportLandmarks[i] = new Vector4(p.x, p.y, p.z, landmarks[i].w);
                if (landmarks[i].w > visibilityThreshold) 
                    draw.Point(p, 0.2f);
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
