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
            viewportLandmarks = new Vector4[PoseMesh.LandmarkCount];
            draw = new PrimitiveDraw(Camera.main, gameObject.layer);
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
            DrawLandmarkResult();
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
            handDetectResults = handDetect.GetResults();
            if (handDetectResults.Count <= 0) return;
            handMesh.Invoke(texture, handDetectResults[0]);
            handMeshResult = handMesh.GetResult();
        }

        private void DetectPose(Texture texture)
        {
            if (texture == null) return;
            poseDetect.Invoke(texture);
            poseDetectResult = poseDetect.GetResults();
            if (poseDetectResult == null) return;
            // poseMeshResult = poseMesh.Invoke(texture, poseDetectResult);
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
        private void DrawLandmarkResult(float visibilityThreshold=0.5f, float zOffset=0f)
        {
            print(poseMeshResult);
            if (poseMeshResult == null)
            {
                return;
            }
        
            // draw.color = Color.blue;
        
            Vector4[] landmarks = poseMeshResult.viewportLandmarks;
            // Update world joints
            for (int i = 0; i < landmarks.Length; i++)
            {
                Vector3 p = GetComponent<Camera>().ViewportToWorldPoint(landmarks[i]);
                viewportLandmarks[i] = new Vector4(p.x, p.y, p.z + zOffset, landmarks[i].w);
            }
        
            // Draw
            for (int i = 0; i < viewportLandmarks.Length; i++)
            {
                Vector4 p = viewportLandmarks[i];
                if (p.w > visibilityThreshold)
                {
                    draw.Cube(p, 0.2f);
                }
            }
            var connections = PoseMesh.Connections;
            for (int i = 0; i < connections.Length; i += 2)
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
