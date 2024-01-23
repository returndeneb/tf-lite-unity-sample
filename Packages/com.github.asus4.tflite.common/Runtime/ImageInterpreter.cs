using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using UnityEngine;

public abstract class ImageInterpreter<T> : System.IDisposable
    where T : struct
{
    public enum Accelerator
    {
        NONE = 0,
        NNAPI = 1,
        GPU = 2,
        XNNPACK = 4,
    }

    protected readonly Interpreter interpreter;
    protected readonly int width;
    protected readonly int height;
    protected readonly int channels;
    protected readonly T[,,] inputTensor;
    protected readonly TextureToTensor tex2Tensor;
    protected readonly TextureResizer resizer;
    protected TextureResizer.ResizeOptions resizeOptions;

    public Texture InputTex
    {
        get
        {
            return (tex2Tensor.texture != null)
                ? tex2Tensor.texture as Texture
                : resizer.outputTexture as Texture;
        }
    }
    public Material TransformMat => resizer.Material;

    public TextureResizer.ResizeOptions ResizeOptions
    {
        get => resizeOptions;
        set => resizeOptions = value;
    }

    public ImageInterpreter(byte[] modelData, InterpreterOptions options)
    {
        try
        {
            interpreter = new Interpreter(modelData, options);
        }
        catch (System.Exception e)
        {
            interpreter?.Dispose();
            throw e;
        }

        interpreter.LogIOInfo();
        {
            var inputShape0 = interpreter.GetInputTensorInfo(0).shape;
            height = inputShape0[1];
            width = inputShape0[2];
            channels = inputShape0[3];
            inputTensor = new T[height, width, channels];

            var inputCount = interpreter.GetInputTensorCount();
            for (var i = 0; i < inputCount; i++)
            {
                var shape = interpreter.GetInputTensorInfo(i).shape;
                interpreter.ResizeInputTensor(i, shape);
            }
            interpreter.AllocateTensors();
        }

        tex2Tensor = new TextureToTensor();
        resizer = new TextureResizer();
        resizeOptions = new TextureResizer.ResizeOptions()
        {
            aspectMode = AspectMode.Fit,
            rotationDegree = 0,
            mirrorHorizontal = false,
            mirrorVertical = false,
            width = width,
            height = height,
        };
    }

    public ImageInterpreter(string modelPath, InterpreterOptions options)
        : this(FileUtil.LoadFile(modelPath), options)
    {
    }

    public ImageInterpreter(string modelPath, Accelerator accelerator)
        : this(modelPath, CreateOptions(accelerator))
    {
    }

    protected static InterpreterOptions CreateOptions(Accelerator accelerator)
    {
        var options = new InterpreterOptions();

        switch (accelerator)
        {
            case Accelerator.NONE:
                options.threads = SystemInfo.processorCount;
                break;
            case Accelerator.NNAPI:
                break;
            case Accelerator.GPU:
                options.AddGpuDelegate();
                break;
            case Accelerator.XNNPACK:
                options.threads = SystemInfo.processorCount;
                options.AddDelegate(XNNPackDelegate.DelegateForType(typeof(T)));
                break;
            default:
                options.Dispose();
                throw new System.NotImplementedException();
        }
        return options;
    }

    public virtual void Dispose()
    {
        interpreter?.Dispose();
        tex2Tensor?.Dispose();
        resizer?.Dispose();
    }

    protected void ToTensor(Texture inputTex, float[,,] inputs)
    {
        var tex = resizer.Resize(inputTex, resizeOptions);
        tex2Tensor.ToTensor(tex, inputs);
    }

    protected void ToTensor(RenderTexture inputTex, float[,,] inputs, bool resize)
    {
        var tex = resize ? resizer.Resize(inputTex, resizeOptions) : inputTex;
        tex2Tensor.ToTensor(tex, inputs);
    }
    
    protected async UniTask<bool> ToTensorAsync(Texture inputTex, float[,,] inputs, CancellationToken cancellationToken)
    {
        var tex = resizer.Resize(inputTex, resizeOptions);
        await tex2Tensor.ToTensorAsync(tex, inputs, cancellationToken);
        return true;
    }

    protected async UniTask<bool> ToTensorAsync(RenderTexture inputTex, float[,,] inputs, bool resize, CancellationToken cancellationToken)
    {
        var tex = resize ? resizer.Resize(inputTex, resizeOptions) : inputTex;
        await tex2Tensor.ToTensorAsync(tex, inputs, cancellationToken);
        return true;
    }
}