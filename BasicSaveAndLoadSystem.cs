using UnityEngine;
using System.Collections;

[System.Serializable]
public class EnvironmentSaveData
{
    public SeasonManagementSystem.Season currentSeason;
    public float climateChange;
    // Add other relevant data
}

public class EnvironmentSaveLoadController : MonoBehaviour
{
    public SeasonManagementSystem SeasonManagementSystem;
    public ClimateChangeController climateChangeController;

    public void SaveEnvironmentSettings()
    {
        EnvironmentSaveData data = new EnvironmentSaveData
        {
            currentSeason = SeasonManagementSystem.currentSeason,
            climateChangeFactor = climateChangeController.GetClimateChangeFactor()
        };

        // Code to save 'data' to a file or database
    }

    public void LoadEnvironmentSettings()
    {
        // Code to load data from a file or database
        EnvironmentSaveData data = /* Load data */
        
        SeasonManagementSystem.SetCurrentSeason(data.currentSeason);
        climateChangeController.SetClimateChangeFactor(data.climateChangeFactor);
    }
}
