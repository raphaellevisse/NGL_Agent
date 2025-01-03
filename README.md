RL AGENT ongoing implementation by Raphael Levisse | PNI December 2024

example_state = """{
  "dimensions": {
    "x": [
      4e-9,
      "m"
    ],
    "y": [
      4e-9,
      "m"
    ],
    "z": [
      4e-8,
      "m"
    ]
  },
  "position": [
    130879.59375,
    78003.6953125,
    1720.1561279296875
  ],
  "crossSectionScale": 1.8496565995583267,
  "projectionOrientation": [
    0.34256511926651,
    -0.1683930605649948,
    -0.6954694390296936,
    -0.6087816953659058
  ],
  "projectionScale": 30260.083367410043,
  "layers": [
    {
      "type": "image",
      "source": "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14",
      "tab": "source",
      "name": "Maryland (USA)-image"
    },
    {
      "type": "segmentation",
      "source": "precomputed://gs://flywire_v141_m783",
      "tab": "source",
      "segments": [
        "720575940623044103"
      ],
      "name": "flywire_v141_m783"
    }
  ],
  "showDefaultAnnotations": false,
  "selectedLayer": {
    "size": 350,
    "visible": true,
    "layer": "flywire_v141_m783"
  },
  "layout": "xz-3d"
}"""