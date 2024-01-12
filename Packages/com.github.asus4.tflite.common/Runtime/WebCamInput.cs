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
    [SerializeField] private Vector2Int requestSize = new Vector2Int(1280, 720);
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
        var tex = NormalizeWebcam(webCamTexture, Screen.width, Screen.height, isFrontFacing);
        onTextureUpdate.Invoke(tex);
    }

    // Invoked by Unity Event
    [Preserve]
    public void ToggleCamera()
    {
        deviceIndex = (deviceIndex + 1) % devices.Length;
        StartCamera(devices[deviceIndex]);
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
        if (webCamTexture == null)
        {
            return;
        }
        webCamTexture.Stop();
        Destroy(webCamTexture);
    }

    private RenderTexture NormalizeWebcam(WebCamTexture texture, int width, int height, bool isFrontFacing)
    {
        var cameraWidth = texture.width;
        var cameraHeight = texture.height;
        var isPortrait = IsPortrait(texture);
        if (isPortrait)
        {
            (cameraWidth, cameraHeight) = (cameraHeight, cameraWidth); // swap
        }

        var cameraAspect = (float)cameraWidth / cameraHeight;
        var targetAspect = (float)width / height;

        int w, h;
        if (cameraAspect > targetAspect)
        {
            w = RoundToEven(cameraHeight * targetAspect);
            h = cameraHeight;
        }
        else
        {
            w = cameraWidth;
            h = RoundToEven(cameraWidth / targetAspect);
        }

        Matrix4x4 mtx;
        Vector4 uvRect;
        var rotation = texture.videoRotationAngle;

        // Seems to be bug in the android. might be fixed in the future.
        if (Application.platform == RuntimePlatform.Android)
        {
            rotation = -rotation;
        }

        if (isPortrait)
        {
            mtx = TextureResizer.GetVertTransform(rotation, texture.videoVerticallyMirrored, isFrontFacing);
            uvRect = TextureResizer.GetTextureSt(targetAspect, cameraAspect, AspectMode.Fill);
        }
        else
        {
            mtx = TextureResizer.GetVertTransform(rotation, isFrontFacing, texture.videoVerticallyMirrored);
            uvRect = TextureResizer.GetTextureSt(cameraAspect, targetAspect, AspectMode.Fill);
        }

        // Debug.Log($"camera: rotation:{texture.videoRotationAngle} flip:{texture.videoVerticallyMirrored}");
        return resizer.Resize(texture, w, h, false, mtx, uvRect);
    }

    private static bool IsPortrait(WebCamTexture texture)
    {
        return texture.videoRotationAngle == 90 || texture.videoRotationAngle == 270;
    }

    private static int RoundToEven(float n)
    {
        return Mathf.RoundToInt(n / 2) * 2;
    }
}