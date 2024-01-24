using System;
using TensorFlowLite;
using UnityEngine;

public class TextureResizer : IDisposable
{
    public struct ResizeOptions
    {
        public int width;
        public int height;
        public AspectMode aspectMode;
    }

    private Material blitMaterial;
    
    public RenderTexture OutputTexture { get; private set; }

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
        set => Material.SetVector(Shader.PropertyToID("_UVRect"), value);
    }

    public Matrix4x4 VertexTransform
    {
        set => Material.SetMatrix(Shader.PropertyToID("_VertTransform"), value);
    }
    

    public void Dispose()
    {
        DisposeUtil.TryDispose(OutputTexture);
        DisposeUtil.TryDispose(blitMaterial);
    }


    public RenderTexture Resize(Texture texture, ResizeOptions options) // 카메라 이미지를 줄일때만 쓰임
    {
        VertexTransform = Matrix4x4.identity; // No rotation
        UVRect = GetTextureSt(texture, options);
        
        if (OutputTexture == null || OutputTexture.width != options.width || OutputTexture.height != options.height)
        {
            DisposeUtil.TryDispose(OutputTexture);
            OutputTexture = new RenderTexture(options.width, options.height, 0, RenderTextureFormat.ARGB32);
        }
        Graphics.Blit(texture, OutputTexture, Material, 0);
        return OutputTexture;
    }

    public RenderTexture Resize(Texture texture, int width, int height, Matrix4x4 transform, Vector4 uvRect)
    {
        VertexTransform = transform;
        UVRect = uvRect;
        if (OutputTexture == null || OutputTexture.width != width || OutputTexture.height != height)
        {
            DisposeUtil.TryDispose(OutputTexture);
            OutputTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32);
        }
        Graphics.Blit(texture, OutputTexture, Material, 0);
        return OutputTexture;
    }
    

    public static Vector4 GetTextureSt(Texture sourceTex, ResizeOptions options)
    {
        var srcAspect = (float)sourceTex.width / sourceTex.height;
        var dstAspect = (float)options.width / options.height;
        var mode = options.aspectMode;
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
            default:
                throw new ArgumentOutOfRangeException();
        }
    }
}