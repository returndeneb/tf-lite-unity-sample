using TensorFlowLite;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Scripting;
using UnityEngine.Serialization;

public sealed class WebCamInput : MonoBehaviour
{
    [System.Serializable]
    public class TextureUpdateEvent : UnityEvent<Texture> { }

    [SerializeField, WebCamName] private string editorCameraName;
    [SerializeField] private WebCamKind preferKind = WebCamKind.WideAngle;
    [SerializeField] private bool isFrontFacing = false;
    [SerializeField] private Vector2Int requestSize = new (1280, 720);
    [SerializeField] private int requestFps = 60;
    public TextureUpdateEvent onTextureUpdate;

    private TextureResizer resizer;
    private WebCamTexture webCamTexture;
    private WebCamDevice[] devices;
    private int deviceIndex;

    private void Start()
    {
        resizer = new TextureResizer();
        devices = WebCamTexture.devices;
        var cameraName = Application.isEditor
            ? editorCameraName
            : WebCamUtil.FindName(preferKind, isFrontFacing);
        WebCamDevice device = default;
        for (var i = 0; i < devices.Length; i++)
        {
            if (devices[i].name != cameraName) continue;
            device = devices[i];
            deviceIndex = i;
            break;
        }
        StartCamera(device);
    }

    private void OnDestroy()
    {
        StopCamera();
        resizer?.Dispose();
    }

    private void Update()
    {
        if (webCamTexture.didUpdateThisFrame) return;
        onTextureUpdate.Invoke(webCamTexture);
    }
    
    private void StartCamera(WebCamDevice device)
    {
        StopCamera();
        isFrontFacing = device.isFrontFacing;
        webCamTexture = new WebCamTexture(device.name, requestSize.x, requestSize.y, requestFps);
        webCamTexture.Play();
    }
    private void StopCamera()
    {
        if (webCamTexture == null) return;
        webCamTexture.Stop();
        Destroy(webCamTexture);
    }
}