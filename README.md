# harp-python

A low-level interface to data collected with the [Harp](https://harp-tech.org/) binary protocol.

## Data model 

The interface makes use of a Pydantic data model generated from Harp device schema definitions. The schema data classes are used to automatically generate binary readers for each device.

All binary data files from a single device need to be stored in the same folder alongside the device meta-schema, named `device.yml`. Each register file should have the following naming convention `<deviceName>_<registerAddress>.bin`.

For example, for a dataset collected with a `Behavior` device, you might have:

```
📦device.harp
 ┣ 📜Behavior_0.bin
 ┣ 📜Behavior_1.bin
...
 ┗ 📜device.yml
```

## How to use

### Create device reader object from schema

```python
import harp
reader = harp.create_reader("device.harp")
```

### Read data from named register

```python
reader.OperationControl.read()
```

### Access register metadata

```python
reader.OperationControl.register.address
```

### Create device reader object with UTC datetime format

```python
reader = harp.create_reader("device.harp", epoch=harp.REFERENCE_EPOCH)
```

### Read data with message type information

```python
reader.OperationControl.read(keep_type=True)
```

### Read data from a specific file

```python
reader.OperationControl.read("data/Behavior_10.bin")
```
