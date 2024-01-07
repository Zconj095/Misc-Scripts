using UnityEngine;

public class EnvironmentalTransitionController : MonoBehaviour
{
    public SeasonManagementSystem SeasonManagementSystem;
    // Environmental factors such as lighting, color, etc.
    public Light directionalLight;
    public Material[] seasonalMaterials; // Array of materials for different seasons

    private float transitionDuration = 10.0f; // Duration of the transition
    private float transitionTimer;
    private bool isTransitioning = false;
    private SeasonManagementSystem.Season fromSeason, toSeason;

    void Start()
    {
        SeasonManagementSystem.OnSeasonChange += StartTransition;
    }

    void OnDestroy()
    {
        SeasonManagementSystem.OnSeasonChange -= StartTransition;
    }

    void Update()
    {
        if (isTransitioning)
        {
            TransitionEnvironment();
        }
    }

    void StartTransition(SeasonManagementSystem.Season newSeason)
    {
        fromSeason = SeasonManagementSystem.currentSeason;
        toSeason = newSeason;
        transitionTimer = transitionDuration;
        isTransitioning = true;
    }

    void TransitionEnvironment()
    {
        transitionTimer -= Time.deltaTime;
        if (transitionTimer <= 0)
        {
            isTransitioning = false;
            // Finalize transition, set environment to 'toSeason' state
            ApplySeasonalChanges(toSeason);
            return;
        }

        float progress = (transitionDuration - transitionTimer) / transitionDuration;
        // Interpolate environmental factors here
        InterpolateLighting(progress);
        InterpolateMaterials(progress);
        // Add more interpolations as needed
    }

    void InterpolateLighting(float progress)
    {
        // Example: Change light intensity or color over time
        // This is a simplified example, adjust as per your game's requirements
        float targetIntensity = /* Calculate based on 'toSeason' */
        directionalLight.intensity = Mathf.Lerp(directionalLight.intensity, targetIntensity, progress);
    }

    void InterpolateMaterials(float progress)
    {
        // Change material properties, like colors, smoothly over time
        Material targetMaterial = seasonalMaterials[(int)toSeason];
        // Apply interpolation to material properties
    }

    void ApplySeasonalChanges(SeasonManagementSystem.Season season)
    {
        // Apply final changes for the new season
    }
}
