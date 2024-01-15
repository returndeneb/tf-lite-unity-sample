using TensorFlowLite;
using UnityEngine;

public class TextureResizer : System.IDisposable
{
    public struct ResizeOptions
    {
        public int width;
        public int height;
        public float rotationDegree;
        public bool mirrorHorizontal;
        public bool mirrorVertical;
        public AspectMode aspectMode;
    }

    private Material blitMaterial;

    static readonly int _VertTransform = Shader.PropertyToID("_VertTransform");
    static readonly int _UVRect = Shader.PropertyToID("_UVRect");

    public RenderTexture outputTexture { get; private set; }

    public Material Material
    {
        get
        {
            blitMaterial ??= new Material(Shader.Find("Hidden/TFLite/Resize"));
            return blitMaterial;
        }
    }

    public Vector4 UVRect
    {
        get => Material.GetVector(_UVRect);
        set => Material.SetVector(_UVRect, value);
    }

    public Matrix4x4 VertexTransform
    {
        get => Material.GetMatrix(_VertTransform);
        set => Material.SetMatrix(_VertTransform, value);
    }
    

    public void Dispose()
    {
        DisposeUtil.TryDispose(outputTexture);
        DisposeUtil.TryDispose(blitMaterial);
    }


    public RenderTexture Resize(Texture texture, ResizeOptions options)
    {
        VertexTransform = GetVertTransform(options.rotationDegree, options.mirrorHorizontal, options.mirrorVertical);
        UVRect = GetTextureST(texture, options);
        return ApplyResize(texture, options.width, options.height);
    }

    public RenderTexture Resize(Texture texture,
        int width, int height,
        Matrix4x4 transform,
        Vector4 uvRect)
    {
        VertexTransform = transform;
        UVRect = uvRect;
        return ApplyResize(texture, width, height);
    }

    private RenderTexture ApplyResize(Texture intputTexture, int outputWidth, int outputHeight)
    {
        if (outputTexture == null || outputTexture.width != outputWidth || outputTexture.height != outputHeight)
        {
            DisposeUtil.TryDispose(outputTexture);
            outputTexture = new RenderTexture(outputWidth, outputHeight, 0, RenderTextureFormat.ARGB32);
        }
        
        Graphics.Blit(intputTexture, outputTexture, Material, 0);
        return outputTexture;
    }

    public static Vector4 GetTextureSt(float srcAspect, float dstAspect, AspectMode mode)
    {
        switch (mode)
        {
            case AspectMode.None:
                return new Vector4(1, 1, 0, 0);
            case AspectMode.Fit:
                if (srcAspect > dstAspect)
                {
                    var s = srcAspect / dstAspect;
                    return new Vector4(1, s, 0, (1 - s) / 2);
                }
                else
                {
                    var s = dstAspect / srcAspect;
                    return new Vector4(s, 1, (1 - s) / 2, 0);
                }
            case AspectMode.Fill:
                if (srcAspect > dstAspect)
                {
                    var s = dstAspect / srcAspect;
                    return new Vector4(s, 1, (1 - s) / 2, 0);
                }
                else
                {
                    var s = srcAspect / dstAspect;
                    return new Vector4(1, s, 0, (1 - s) / 2);
                }
        }
        throw new System.Exception("Unknown aspect mode");
    }

    public static Vector4 GetTextureST(Texture sourceTex, ResizeOptions options)
    {
        return GetTextureSt(
            (float)sourceTex.width / sourceTex.height, // src
            (float)options.width / options.height, // dst
            options.aspectMode);
    }

    private static readonly Matrix4x4 PushMatrix = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
    private static readonly Matrix4x4 PopMatrix = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));
    internal static Matrix4x4 GetVertTransform(float rotation, bool mirrorHorizontal, bool mirrorVertical)
    {
        var scale = new Vector3(
            mirrorHorizontal ? -1 : 1,
            mirrorVertical ? -1 : 1,
            1);
        var trs = Matrix4x4.TRS(
            Vector3.zero,
            Quaternion.Euler(0, 0, rotation),
            scale
        );
        return PushMatrix * trs * PopMatrix;
    }

}