using System.Collections.Generic;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// BlazeFace from MediaPile
/// https://github.com/google/mediapipe
/// https://viz.mediapipe.dev/demo/face_detection
/// </summary>
[RequireComponent(typeof(WebCamInput))]
public class FaceDetectionSample : MonoBehaviour
{

    [SerializeField]
    private RawImage cameraView;

    private FaceDetect faceDetect;
    private List<FaceDetect.Result> results;
    private PrimitiveDraw draw;
    private readonly Vector3[] rtCorners = new Vector3[4];

    private void Start()
    {
        faceDetect = new FaceDetect("mediapipe/face_detection_back.tflite");
        draw = new PrimitiveDraw(Camera.main, gameObject.layer);
        draw.color = Color.blue;
        GetComponent<WebCamInput>().OnTextureUpdate.AddListener(OnTextureUpdate);
        cameraView.material = faceDetect.transformMat;
        cameraView.rectTransform.GetWorldCorners(rtCorners);
    }

    private void OnDestroy()
    {
        GetComponent<WebCamInput>().OnTextureUpdate.RemoveListener(OnTextureUpdate);
        faceDetect?.Dispose();
        draw?.Dispose();
    }

    private void Update()
    {
        DrawResults(results);
    }

    private void OnTextureUpdate(Texture texture)
    {
        faceDetect.Invoke(texture);
        results = faceDetect.GetResults();
    }

    private void DrawResults(List<FaceDetect.Result> results)
    {
        if (results == null || results.Count == 0) return;
        foreach (var result in results)
        {
            Rect rect = MathTF.Lerp(rtCorners[0], rtCorners[2], result.rect, true);
            draw.Rect(rect, 0.05f);
            foreach (Vector2 p in result.keypoints)
            {
                draw.Point(MathTF.Lerp(rtCorners[0], rtCorners[2], new Vector3(p.x, 1f - p.y, 0)), 0.1f);
            }
        }
        draw.Apply();
    }
}
