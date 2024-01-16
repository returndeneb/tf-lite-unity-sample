using System.Linq;
using Samples.FaceMesh;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(WebCamInput))]
public sealed class FaceMeshSample : MonoBehaviour
{

    [SerializeField]
    private bool useLandmarkToDetection = true;

    [SerializeField]
    private RawImage cameraView = null;

    [SerializeField]
    private RawImage croppedView = null;

    [SerializeField]
    private Material faceMaterial = null;

    private FaceDetect faceDetect;
    private FaceMesh faceMesh;
    private PrimitiveDraw draw;
    private MeshFilter faceMeshFilter;
    private Vector3[] faceKeypoints;
    private FaceDetect.Result detectionResult;
    private FaceMesh.Result meshResult;
    private readonly Vector3[] rtCorners = new Vector3[4];

    private void Start()
    {
        faceDetect = new FaceDetect("mediapipe/face_detection_back.tflite");
        faceMesh = new FaceMesh("mediapipe/face_landmark.tflite");
        draw = new PrimitiveDraw(Camera.main, gameObject.layer);
        cameraView.material = faceDetect.TransformMat;
        {
            var go = new GameObject("Face");
            go.transform.SetParent(transform);
            var faceRenderer = go.AddComponent<MeshRenderer>();
            faceRenderer.material = faceMaterial;

            faceMeshFilter = go.AddComponent<MeshFilter>();
            faceMeshFilter.sharedMesh = FaceMeshBuilder.CreateMesh();

            faceKeypoints = new Vector3[FaceMesh.KEYPOINT_COUNT];
        }

        GetComponent<WebCamInput>().onTextureUpdate.AddListener(OnTextureUpdate);
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
        DrawResults(detectionResult, meshResult);
    }

    private void OnTextureUpdate(Texture texture)
    {
        if (detectionResult == null || !useLandmarkToDetection)
        {
            faceDetect.Invoke(texture);
            cameraView.texture = texture;
            detectionResult = faceDetect.GetResults().FirstOrDefault();

            if (detectionResult == null)
            {
                return;
            }
        }

        faceMesh.Invoke(texture, detectionResult);
        croppedView.texture = faceMesh.InputTex;
        meshResult = faceMesh.GetResult();

        if (meshResult.score < 0.5f)
        {
            detectionResult = null;
            return;
        }

        if (useLandmarkToDetection)
        {
            detectionResult = faceMesh.LandmarkToDetection(meshResult);
        }
    }

    private void DrawResults(FaceDetect.Result detection, FaceMesh.Result face)
    {
        cameraView.rectTransform.GetWorldCorners(rtCorners);
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        // Draw Face Detection
        if (detection != null)
        {
            draw.color = Color.blue;
            Rect rect = MathTF.Lerp(min, max, detection.rect, true);
            draw.Rect(rect, 0.05f);
            foreach (Vector2 p in detection.keypoints)
            {
                draw.Point(MathTF.Lerp(min, max, new Vector3(p.x, 1f - p.y, 0)), 0.1f);
            }
            draw.Apply();
        }

        if (face != null)
        {
            // Draw face
            draw.color = Color.green;
            var zScale = (max.x - min.x) / 2;
            for (int i = 0; i < face.keypoints.Length; i++)
            {
                Vector3 kp = face.keypoints[i];
                kp.y = 1f - kp.y;
                print(kp.y);
                print(1f-kp.y);

                Vector3 p = MathTF.Lerp(min, max, kp);
                p.z = face.keypoints[i].z * zScale;

                faceKeypoints[i] = p;
                draw.Point(p, 0.05f);
            }
            draw.Apply();

            // Update Mesh
            FaceMeshBuilder.UpdateMesh(faceMeshFilter.sharedMesh, faceKeypoints);
        }
    }
}
