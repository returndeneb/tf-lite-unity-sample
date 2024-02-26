using TensorFlowLite;
using UnityEngine;
using UnityEngine.Events;

public sealed class WebCamInput : MonoBehaviour
{
    [System.Serializable] public class TextureUpdateEvent : UnityEvent<Texture> {}
    [SerializeField,WebCamName] private string webcamName;
    public TextureUpdateEvent onTextureUpdate;
    private WebCamTexture webCamTexture;

    private void Start()
    {
        WebCamDevice device = default;
        foreach (var webcam in WebCamTexture.devices)
        {
            if (webcam.name != webcamName) continue;
            device = webcam;
            break;
        }
        StartCamera(device);
    }
    private void OnDestroy()
    {
        StopCamera();
    }
    private void Update()
    {
        if (webCamTexture.didUpdateThisFrame) return;
        onTextureUpdate.Invoke(webCamTexture);
    }
    private void StartCamera(WebCamDevice device)
    {
        StopCamera();
        webCamTexture = new WebCamTexture(device.name, 1280, 720, 60);
        webCamTexture.Play();
    }
    private void StopCamera()
    {
        if (webCamTexture == null) return;
        webCamTexture.Stop();
        Destroy(webCamTexture);
    }
}