using UnityEngine;

public class TheSunlightController : MonoBehaviour
{
    public SeasonManagementSystem SeasonManagementSystem;
    public Light sunLight;

    void Update()
    {
        AdjustSunlight();
    }

    void AdjustSunlight()
    {
        switch (SeasonManagementSystem.currentSeason)
        {
            case SeasonManagementSystem.Season.Spring:
            case SeasonManagementSystem.Season.Autumn:
                // Neutral angle and intensity
                sunLight.transform.rotation = Quaternion.Euler(45, 0, 0);
                sunLight.intensity = 1.0f;
                break;
            case SeasonManagementSystem.Season.Summer:
                // Higher angle and intensity
                sunLight.transform.rotation = Quaternion.Euler(60, 0, 0);
                sunLight.intensity = 1.2f;
                break;
            case SeasonManagementSystem.Season.Winter:
                // Lower angle and intensity
                sunLight.transform.rotation = Quaternion.Euler(30, 0, 0);
                sunLight.intensity = 0.8f;
                break;
        }
    }
}
