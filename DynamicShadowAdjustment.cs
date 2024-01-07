using UnityEngine;

public class DynamicShadowController : MonoBehaviour
{
    public SeasonManagementSystem SeasonManagementSystem;
    public Light sunLight; // The main directional light

    void Update()
    {
        AdjustShadowSettings();
    }

    void AdjustShadowSettings()
    {
        // Example: Lower shadow distance during winter due to lower sun angle
        if (SeasonManagementSystem.currentSeason == SeasonManagementSystem.Season.Winter)
        {
            QualitySettings.shadowDistance = 100; // Adjust based on need
            // Other potential adjustments: shadow resolution, shadow projection
        }
        else
        {
            QualitySettings.shadowDistance = 200; // Default or summer value
        }

        // Additional logic can be added for time of day adjustments
    }
}
