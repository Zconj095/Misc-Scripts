using UnityEngine;

public class VegetationController : MonoBehaviour
{
    public SeasonManagementSystem SeasonManagementSystem;
    // Assume these materials are set in the inspector to match the seasons
    public Material springMaterial;
    public Material summerMaterial;
    public Material autumnMaterial;
    public Material winterMaterial;

    void Start()
    {
        SeasonManagementSystem.OnSeasonChange += UpdateVegetation;
    }

    void OnDestroy()
    {
        SeasonManagementSystem.OnSeasonChange -= UpdateVegetation;
    }

    void UpdateVegetation(SeasonManagementSystem.Season newSeason)
    {
        switch (newSeason)
        {
            case SeasonManagementSystem.Season.Spring:
                ChangeVegetation(springMaterial);
                break;
            case SeasonManagementSystem.Season.Summer:
                ChangeVegetation(summerMaterial);
                break;
            case SeasonManagementSystem.Season.Autumn:
                ChangeVegetation(autumnMaterial);
                break;
            case SeasonManagementSystem.Season.Winter:
                ChangeVegetation(winterMaterial);
                break;
        }
    }

    void ChangeVegetation(Material newMaterial)
    {
        // Logic to change the vegetation's material
        // This is a simplified example. You may need to adjust various properties 
        // and might have multiple objects to update.
    }
}
