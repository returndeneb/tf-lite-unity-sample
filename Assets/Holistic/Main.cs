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
        
        private PoseDetect poseDetect;
        private PoseMesh poseMesh;
        private PoseDetect.Result poseDetectResult;
        private PoseMesh.Result poseMeshResult;
        
        private FaceDetect faceDetect;
        private FaceMesh faceMesh;
        private IrisMesh irisMesh;
        private FaceDetect.Result faceDetectResult;
        private FaceMesh.Result faceMeshResult;
        private IrisMesh.Result irisMeshResult;
        
        private HandDetect handDetect;
        private HandMesh handMesh;
        private HandMesh handMesh2;
        private List<HandDetect.Result> handDetectResults;
        private HandMesh.Result handMeshResult;
        private HandMesh.Result handMeshResult2;
        
        private PrimitiveDraw draw;
        private readonly Vector3[] imgSize = new Vector3[4];
        private Vector4[] viewportLandmarks;
        private void Start()
        {
            poseDetect = new PoseDetect("pose_detection.tflite");
            poseMesh = new PoseMesh("pose_landmark_lite.tflite");
            faceDetect = new FaceDetect("face_detection.tflite");
            faceMesh = new FaceMesh("face_landmark.tflite");
            irisMesh = new IrisMesh("iris_landmark.tflite");
            handDetect = new HandDetect("palm_detection.tflite");
            handMesh = new HandMesh("hand_landmark.tflite");
            handMesh2 = new HandMesh("hand_landmark.tflite");
            
            
            
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
            // DetectHand(texture);
            // DetectPose(texture);
        }
        private void Update()
        {
            image.rectTransform.GetWorldCorners(imgSize);
            DrawFace();
            // DrawHand();
            // DrawPose();
            
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
            irisMesh.Invoke(texture, faceMeshResult);
            irisMeshResult = irisMesh.GetResult();
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
                var p = MathTF.Lerp(imgSize[0], imgSize[2], kp, true);
                p.z = faceMeshResult.keyPoints[i].z * (imgSize[2].x - imgSize[0].x) / 2;

                if (i == 33 || i == 133 || i == 362 || i == 263)
                {
                    draw.color = Color.red;
                }
                else
                {
                    draw.color = Color.green;
                }
                draw.Point(p);
                draw.Apply();
            }

            for (var i = 0; i < irisMeshResult.keyPoints.Length; i++)
            {
                var kp = irisMeshResult.keyPoints[i];
                var p = MathTF.Lerp(imgSize[0], imgSize[2], kp, false);
                
                draw.color = Color.yellow;
                draw.Point(p);
                draw.Apply();
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
                var p = MathTF.Lerp(imgSize[0],imgSize[2],landmarks[i],true);
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
