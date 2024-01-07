using UnityEngine;

public class MoonController : MonoBehaviour
{
	public Light moonLight;
	public Material[] moonPhaseMaterials; // Materials for different moon phases
	private int currentMoonPhase = 0; // Index for moon phases

	private float moonPhaseDuration = 5.0f; // Duration of each moon phase in game time
	private float moonPhaseTimer;

	void Start()
	{
		moonPhaseTimer = moonPhaseDuration;
		UpdateMoonAppearance();
	}

	void Update()
	{
		moonPhaseTimer -= Time.deltaTime;
		if (moonPhaseTimer <= 0)
		{
			ChangeMoonPhase();
			moonPhaseTimer = moonPhaseDuration;
		}
	}

	void ChangeMoonPhase()
	{
		currentMoonPhase = (currentMoonPhase + 1) % moonPhaseMaterials.Length;
		UpdateMoonAppearance();
	}

	void UpdateMoonAppearance()
	{
		// Change moon material to reflect current phase
		GetComponent<Renderer>().material = moonPhaseMaterials[currentMoonPhase];

		// Adjust moonlight intensity based on phase, e.g., dimmer during new moon
		moonLight.intensity = CalculateMoonlightIntensity(currentMoonPhase);
	}

	float CalculateMoonlightIntensity(int moonPhase)
	{
		// Implement logic to determine moonlight intensity based on phase
		// For example, full moon has higher intensity than a new moon
		// This is a placeholder function
		return 0.5f + 0.5f * moonPhase / (moonPhaseMaterials.Length - 1);
	}
}
