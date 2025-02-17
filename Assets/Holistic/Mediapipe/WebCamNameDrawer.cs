﻿using System.Linq;
using UnityEditor;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// A custom attribute that enables to select webcam name from the popup.
    /// </summary>
    [CustomPropertyDrawer(typeof(WebCamName))]
    public class WebCamNameDrawer : PropertyDrawer
    {
        private string[] displayNames = null;
        private int selectedIndex = -1;

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            if (property.propertyType != SerializedPropertyType.String)
            {
                Debug.LogError($"type: {property.propertyType} is not supported.");
                EditorGUI.LabelField(position, label.text, "Use WebcamName with string.");
                return;
            }

            displayNames ??= WebCamTexture.devices.Select(device => device.name).ToArray();

            if (selectedIndex < 0)
            {
                selectedIndex = FindSelectedIndex(displayNames, property.stringValue);
            }

            EditorGUI.BeginProperty(position, label, property);

            selectedIndex = EditorGUI.Popup(position, label.text, selectedIndex, displayNames);
            property.stringValue = displayNames[selectedIndex];

            EditorGUI.EndProperty();
        }

        private static int FindSelectedIndex(string[] displayNames, string value)
        {
            for (var i = 0; i < displayNames.Length; i++)
            {
                if (displayNames[i] == value)
                {
                    return i;
                }
            }
            return 0;
        }
    }
}
