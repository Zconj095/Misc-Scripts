using UnityEngine;

public class WeatherController : MonoBehaviour
{
    public SeasonManagementSystem SeasonManagementSystem;
    // Particle systems for different weather types, assigned via inspector
    public ParticleSystem rainParticleSystem;
    public ParticleSystem snowParticleSystem;

    void Start()
    {
        SeasonManagementSystem.OnSeasonChange += UpdateWeather;
    }

    void OnDestroy()
    {
        SeasonManagementSystem.OnSeasonChange -= UpdateWeather;
    }

    void UpdateWeather(SeasonManagementSystem.Season newSeason)
    {
        switch (newSeason)
        {
            case SeasonManagementSystem.Season.Spring:
                StartRain();
                break;
            case SeasonManagementSystem.Season.Winter:
                StartSnow();
                break;
            // Add cases for other seasons as needed
        }
    }

    void StartRain()
    {
        // Logic to start rain particle system
    }

    void StartSnow()
    {
        // Logic to start snow particle system
    }
}
