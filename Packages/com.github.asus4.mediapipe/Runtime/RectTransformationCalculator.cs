using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// RectTransformationCalculator from MediaPipe
    /// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/rect_transformation_calculator.cc
    /// </summary>
    public class RectTransformationCalculator
    {
        public ref struct Options
        {
            /// <summary>
            /// The Normalized Rect (0, 0, 1, 1)
            /// </summary>
            public Rect rect;
            public float rotationDegree;
            public Vector2 shift;
            public Vector2 scale;
        }

        private static readonly Matrix4x4 PopMatrix = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        private static readonly Matrix4x4 PushMatrix = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));

        public static Matrix4x4 CalcMatrix(Options options)
        {
            var rotation = Quaternion.Euler(0, 0, options.rotationDegree);
            var size = Vector2.Scale(options.rect.size, options.scale); // elementwise product
            // Calc center position
            var center = options.rect.center - new Vector2(0.5f, 0.5f);

            center = rotation * center;
            center = center/size+options.shift;

            var trs = Matrix4x4.TRS(
                new Vector3(-center.x, -center.y, 0),
                rotation,
                new Vector3(1 / size.x, 1 / size.y, 1)
            );
            
            return PopMatrix * trs  * PushMatrix;
        }
    }
}
