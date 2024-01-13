// EnviroAudioModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroAudioModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroAudioModule : public UObject
{
	GENERATED_BODY()
	
public:	
	// Properties
	UPROPERTY()
	bool ShowModuleInspector;

	UPROPERTY()
	bool ShowAudioControls;

	// Methods
	UFUNCTION()
	void CreateAudio();

	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroAudioPreset* Preset);
	
};

// EnviroAudioModuleEditor.h 
#pragma once

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroAudioModuleEditor : public FEditor
{
public:
	virtual void OnEnable() override;
	virtual void OnInspectorGUI() override;
	
private:
	UEnviroAudioModule* AudioModule;
	
};



// EnviroAuroraModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroAuroraModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroAuroraModule : public UObject  
{
	GENERATED_BODY()

public:

	// Properties
	UPROPERTY(EditAnywhere)
	bool bUseAurora;

	UPROPERTY(EditAnywhere)	
	float AuroraIntensity;

	// Methods
	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroAuroraPreset* Preset);

};

// EnviroAuroraModuleEditor.h
#pragma once  

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroAuroraModuleEditor : public FEditor
{
public:
	virtual void OnEnable() override;

	virtual void OnInspectorGUI() override;
	
private:
	UEnviroAuroraModule* AuroraModule;

	UPROPERTY()
	UEnviroAuroraPreset* AuroraPreset;
	
};


// EnviroDefaultModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroDefaultModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroDefaultModule : public UObject
{
	GENERATED_BODY()
	
public:

	// Properties

	// Methods
	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroDefaultPreset* Preset);

};

// EnviroDefaultModuleEditor.h
#pragma once

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroDefaultModuleEditor : public FEditor
{
public:
	virtual void OnEnable() override;

	virtual void OnInspectorGUI() override;

private:
	UEnviroDefaultModule* DefaultModule;
	
	UPROPERTY()
	UEnviroDefaultPreset* DefaultPreset;
};

// EnviroDefaultModule.h

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroDefaultModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroDefaultModule : public UObject
{
	GENERATED_BODY()

public:	

	UPROPERTY(EditAnywhere)
	FEnviroGradient FrontColorGradient;
		
};


// EnviroGradient.h

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroGradient.generated.h"

USTRUCT()
struct FEnviroGradient
{
    UPROPERTY(EditAnywhere)
    FLinearColor Color0;
    
    UPROPERTY(EditAnywhere)
    FLinearColor Color1;    
};

// EnviroGradientDetailsCustomization.h

#pragma once

#include "CoreMinimal.h"
#include "IDetailCustomization.h"

class FEnviroGradientDetailsCustomization : public IDetailCustomization
{
public:
    static TSharedRef<IDetailCustomization> MakeInstance();

    virtual void CustomizeDetails(IDetailLayoutBuilder& DetailLayout) override;
    
private:
   IDetailLayoutBuilder* MyDetailLayout;
};

// EnviroEffectsModule.h
#pragma once

#include "CoreMinimal.h"  
#include "UObject/NoExportTypes.h"
#include "EnviroEffectsModule.generated.h"

USTRUCT()
struct FEnviroEffectType 
{
  UPROPERTY()
  FString Name;

  UPROPERTY()
  TSubclassOf<AActor> EffectClass;
  
  // Other properties  
};

UCLASS()
class ENVIRO_API UEnviroEffectsModule : public UObject
{
	GENERATED_BODY()
	
public:
  UPROPERTY()
  TArray<FEnviroEffectType> Effects;
  
  UFUNCTION()
  void CreateEffects();

  UFUNCTION()
  void LoadModuleValues();

  UFUNCTION()
  void SaveModuleValues(UEnviroEffectsPreset* Preset);
  
};

// EnviroEffectsModuleEditor.h
#pragma once

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"  

class FEnviroEffectsModuleEditor : public FEditor
{
public:
  virtual void OnEnable() override;

  virtual void OnInspectorGUI() override;

private:
  UEnviroEffectsModule* EffectsModule;
  
  UPROPERTY()
  UEnviroEffectsPreset* EffectsPreset;
  
};

// EnviroEnvironmentModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroEnvironmentModule.generated.h"

UENUM()
enum class EEnviroSeason : uint8 {
  Spring, Summer, Autumn, Winter  
};

USTRUCT()
struct FEnviroSeasonDates {
  UPROPERTY()
  int32 StartDay;
  
  UPROPERTY()
  int32 EndDay;  
};

UCLASS()
class ENVIRO_API UEnviroEnvironmentModule : public UObject
{
	GENERATED_BODY()
	
public:

  UPROPERTY(EditAnywhere)
  EEnviroSeason CurrentSeason;
  
  UPROPERTY(EditAnywhere) 
  float Temperature;

  UPROPERTY(EditAnywhere)
  FEnviroSeasonDates SpringDates;

  UFUNCTION()
  void LoadModuleValues();

  UFUNCTION()
  void SaveModuleValues(UEnviroEnvironmentPreset* Preset);

};


// EnviroEnvironmentModuleEditor.h
#pragma once  

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroEnvironmentModuleEditor : public FEditor
{
public:
  virtual void OnEnable() override;

  virtual void OnInspectorGUI() override;

private:
  UEnviroEnvironmentModule* EnvironmentModule;
  
  UPROPERTY()
  UEnviroEnvironmentPreset* EnvironmentPreset;
  
};

// EnviroFlatCloudsModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroFlatCloudsModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroFlatCloudsModule : public UObject
{
	GENERATED_BODY()
	
public:
	
	UPROPERTY(EditAnywhere)
	UTexture2D* BaseCloudTexture;

	UPROPERTY(EditAnywhere)
	FLinearColor CloudLightColor;

	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION() 
	void SaveModuleValues(UEnviroFlatCloudsPreset* Preset);

};


// EnviroFlatCloudsModuleEditor.h
#pragma once

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroFlatCloudsModuleEditor : public FEditor
{
public:
	virtual void OnEnable() override;

	virtual void OnInspectorGUI() override;

private:
	UEnviroFlatCloudsModule* FlatCloudsModule;
	
	UPROPERTY()
	UEnviroFlatCloudsPreset* FlatCloudsPreset;
};

// EnviroFogModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroFogModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroFogModule : public UObject
{
	GENERATED_BODY()
	
public:

	UPROPERTY(EditAnywhere)
	bool bEnableFog;

	UPROPERTY(EditAnywhere)
	FLinearColor FogColor;
	
	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroFogPreset* Preset);

};


// EnviroFogModuleEditor.h
#pragma once

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"  

class FEnviroFogModuleEditor : public FEditor
{
public:
	virtual void OnEnable() override;

	virtual void OnInspectorGUI() override;

private:
	UEnviroFogModule* FogModule;
	
	UPROPERTY()
	UEnviroFogPreset* FogPreset;
	
};

// EnviroLightingModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroLightingModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroLightingModule : public UObject  
{
	GENERATED_BODY()
	
public:

	UPROPERTY(EditAnywhere)
	FLinearColor SunLightColor;

	UPROPERTY(EditAnywhere)
	UCurveFloat* SunIntensityCurve;

	UFUNCTION()
	void ApplyLightingChanges();

	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroLightingPreset* Preset);

};

// EnviroLightingModuleEditor.h
#pragma once

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"  

class FEnviroLightingModuleEditor : public FEditor
{
public:
	virtual void OnEnable() override;

	virtual void OnInspectorGUI() override;

private:
	UEnviroLightingModule* LightingModule;
	
	UPROPERTY()
	UEnviroLightingPreset* LightingPreset;
	
};

// EnviroLightningModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroLightningModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroLightningModule : public UObject
{
	GENERATED_BODY()
	
public:

	UPROPERTY(EditAnywhere)
	TSubclassOf<AActor> LightningEffectClass;

	UPROPERTY(EditAnywhere)
	bool bEnableRandomStorms;

	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroLightningPreset* Preset);

};


// EnviroLightningModuleEditor.h
#pragma once  

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroLightningModuleEditor : public FEditor
{
public:
	virtual void OnEnable() override;

	virtual void OnInspectorGUI() override;

private:
	UEnviroLightningModule* LightningModule;
	
	UPROPERTY()
	UEnviroLightningPreset* LightningPreset;
};

// Add a lightning effect UObject
UCLASS()
class ENVIRO_API UEnviroLightningEffect : public UObject
{
	GENERATED_BODY()
	
public:	
	UPROPERTY(EditAnywhere)
	USoundBase* ThunderSound;

	UPROPERTY(EditAnywhere)
	UParticleSystem* LightningEffect;
	
	UPROPERTY(EditAnywhere)
	float DelayBetweenStrikes;
};

// Reference it in the lightning module
UCLASS()
class ENVIRO_API UEnviroLightningModule : public UObject
{

	UPROPERTY(EditAnywhere)
	TArray<UEnviroLightningEffect> LightningEffects;
	
	UFUNCTION()
	void SpawnLightningStrike(const FVector& Location);
	
};

// Spawn strikes from BP in editor
class FEnviroLightningModuleEditor : public FEditor
{
public:

	virtual void OnInspectorGUI() override {
		
		// Button to spawn test strike
		if(GUILayout.Button("Spawn Test Strike"))
		{
			TargetLightningModule->SpawnLightningStrike(GEditor->GetEditorWorldLocation());	
		}
		
	}
	
private:

	UEnviroLightningModule* TargetLightningModule;
	
};

// EnviroQualitySettings.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroQualitySettings.generated.h"

USTRUCT()
struct FEnviroQualityCloudsSettings
{
    UPROPERTY(EditAnywhere)
    bool bEnable;
    
    UPROPERTY(EditAnywhere)
    int32 SamplesPerRay;
    
};

UCLASS(Abstract, EditInlineNew)
class ENVIRO_API UEnviroQualitySettings : public UObject
{
    GENERATED_BODY()
    
public:

    UPROPERTY(EditAnywhere)
    FEnviroQualityCloudsSettings Clouds;

    UPROPERTY(EditAnywhere)
    FEnviroQualityFogSettings Fog;
    
};

UCLASS(Blueprintable)
class ENVIRO_API UEnviroQualityProfile : public UEnviroQualitySettings
{
    GENERATED_BODY()
    
public:
  
    UPROPERTY(EditAnywhere)
    FString ProfileName;
  
};

// EnviroQualityModule.h
#pragma once

#include "CoreMinimal.h"
#include "EnviroQualitySettings.h" 

UCLASS()
class ENVIRO_API UEnviroQualityModule : public UObject
{
    GENERATED_BODY()
    
public:

    UPROPERTY(EditAnywhere)
    TArray<UEnviroQualityProfile> QualityProfiles;

    UPROPERTY(EditAnywhere)
    UEnviroQualityProfile* DefaultProfile;
  
};

// EnviroReflectionProbe.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "EnviroReflectionProbe.generated.h"

UCLASS()
class ENVIRO_API AEnviroReflectionProbe : public AActor
{
	GENERATED_BODY()
	
public:	

	// Properties

	UPROPERTY(EditAnywhere)
	bool bStandaloneProbe;

	UPROPERTY(EditAnywhere)
	bool bCustomRendering;
	
	// Methods

	UFUNCTION(BlueprintCallable)
	void UpdateProbe();
	
};


// EnviroReflectionProbeDetailsCustomization.h
#pragma once

#include "CoreMinimal.h"
#include "IDetailCustomization.h"

class FEnviroReflectionProbeDetails : public IDetailCustomization
{
public:
	static TSharedRef<IDetailCustomization> MakeInstance();

	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;
	
};

// EnviroReflectionsModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroReflectionsModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroReflectionsModule : public UObject
{
	GENERATED_BODY()
	
public:

	UPROPERTY(EditAnywhere)
	float GlobalReflectionsIntensity;

	UPROPERTY(EditAnywhere) 
	TEnumAsByte<ETextureResolution> CubemapResolution;
	
	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroReflectionsPreset* Preset);

};

// EnviroReflectionsModuleEditor.h
#pragma once  

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroReflectionsModuleEditor : public FEditor
{
public:
	virtual void OnEnable() override;

	virtual void OnInspectorGUI() override;

private:
	UEnviroReflectionsModule* ReflectionsModule;
	
	UPROPERTY()
	UEnviroReflectionsPreset* ReflectionsPreset;
	
};

// EnviroSkyModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroSkyModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroSkyModule : public UObject
{
	GENERATED_BODY()
	
public:

	UPROPERTY(EditAnywhere)
	UTextureCube* SkyCubeMap;

	UPROPERTY(EditAnywhere)
	FLinearColor HorizonColor;
	
	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroSkyPreset* Preset);

};

// EnviroSkyModuleEditor.h
#pragma once

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroSkyModuleEditor : public FEditor
{
public:
	virtual void OnEnable() override;

	virtual void OnInspectorGUI() override;
	
private:
	UEnviroSkyModule* SkyModule;
	
	UPROPERTY()
	UEnviroSkyPreset* SkyPreset;
};

// EnviroTimeModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroTimeModule.generated.h"

USTRUCT()
struct FEnviroDateTime
{
    UPROPERTY(EditAnywhere)
    int32 Year;

    UPROPERTY(EditAnywhere) 
    int32 Month;
    
    // Other fields
};

UCLASS()
class ENVIRO_API UEnviroTimeModule : public UObject
{
	GENERATED_BODY()
	
public:

	UPROPERTY(EditAnywhere)
	FEnviroDateTime DateTime;

	UPROPERTY(EditAnywhere)
	float DayNightCycleLength;
	
	UFUNCTION()
	void UpdateModule();

	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroTimePreset* Preset);

};

// EnviroTimeModuleEditor.h
#pragma once

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroTimeModuleEditor : public FEditor 
{
public:
   // Code
};

// EnviroCloudsModule.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroCloudsModule.generated.h"

USTRUCT()
struct FEnviroCloudLayerSettings {

  UPROPERTY(EditAnywhere)
  float CloudDensity;
  
  UPROPERTY(EditAnywhere)
  float CloudCoverage;
  
  // Other properties
  
};

UCLASS()
class ENVIRO_API UEnviroCloudsModule : public UObject
{
	GENERATED_BODY()
	
public:

	UPROPERTY(EditAnywhere)
	FEnviroCloudLayerSettings CloudLayer1;

	UPROPERTY(EditAnywhere)
	FEnviroCloudLayerSettings CloudLayer2;
	
	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroCloudsPreset* Preset);

};

// EnviroCloudsModuleEditor.h
#pragma once

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroCloudsModuleEditor : public FEditor
{
public:
	virtual void OnEnable() override;

	virtual void OnInspectorGUI() override;
	
private:
	UEnviroCloudsModule* CloudsModule;
	
	UPROPERTY()
	UEnviroCloudsPreset* CloudsPreset;
	
};

// EnviroWeatherType.h
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "EnviroWeatherType.generated.h"

UCLASS(Blueprintable)
class ENVIRO_API UEnviroWeatherType : public UObject
{
	GENERATED_BODY()
	
public:

	UPROPERTY(EditAnywhere)
	FString WeatherName;

	UPROPERTY(EditAnywhere) 
	float CloudDensityModifier;
	
	// Other properties
	
};

// EnviroWeatherModule.h 
#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"  
#include "EnviroWeatherType.h"
#include "EnviroWeatherModule.generated.h"

UCLASS()
class ENVIRO_API UEnviroWeatherModule : public UObject
{
	GENERATED_BODY()
	
public:

	UPROPERTY(EditAnywhere)
	TArray<UEnviroWeatherType*> WeatherTypes;

	UFUNCTION()
	void LoadModuleValues();

	UFUNCTION()
	void SaveModuleValues(UEnviroWeatherPreset* Preset);

};

// EnviroWeatherModuleEditor.h
#pragma once

#include "CoreMinimal.h"
#include "Editor/EditorEngine.h"

class FEnviroWeatherModuleEditor : public FEditor
{
   // Editor code
};

// EnviroZone.h
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "EnviroZone.generated.h"

UCLASS()
class ENVIRO_API AEnviroZone : public AActor
{
	GENERATED_BODY()
	
public:

	// Properties

	UPROPERTY(EditAnywhere)
	TArray<UEnviroWeatherType*> AvailableWeatherTypes;

	UPROPERTY(EditAnywhere)
	bool bAutoChangeWeather;

	// Methods

	UFUNCTION(BlueprintCallable)
	void ChangeZoneWeatherInstant(UEnviroWeatherType* NewWeatherType);
		
};

// EnviroZoneDetailsCustomization.h
#pragma once

#include "CoreMinimal.h"
#include "IDetailCustomization.h"

class FEnviroZoneDetails : public IDetailCustomization
{
public:
	static TSharedRef<IDetailCustomization> MakeInstance();

	virtual void CustomizeDetails(IDetailLayoutBuilder& DetailBuilder) override;
	
};

