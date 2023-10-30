# harp-python

A low-level interface to data collected with the [Harp](https://harp-tech.org/) binary protocol.

## Data model 

To regenerate Pydantic data models from device schema definitions, activate a virtual environment with `dev` dependencies, and run:

```
datamodel-codegen --input ./reflex-generator/schema/device.json --output harp/model.py --output-model-type pydantic_v2.BaseModel
```

> [!IMPORTANT]
> Currently code generation adds an unwanted field at the very end of the data model definition `registers: Optional[Any] = None`. This declaration needs to be removed for serialization to work properly.
