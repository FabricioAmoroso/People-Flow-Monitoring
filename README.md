# People-Flow-Monitoring

<img src="https://i.imgur.com/gVSsYe8.png" width="250" height="250">

*Read this in other languages: [English](https://github.com/FabricioAmoroso/People-Flow-Monitoring/blob/master/README.md), [Portuguese](https://github.com/FabricioAmoroso/People-Flow-Monitoring/blob/master/README-pt.md).*

## Project operation

### Description

The following project has the purpose to monitor indoor environments in order to detect people flow.
There are three main features: Area of Interest detection, Social Distancing monitorng and Real Time People Counting.

- Area of Interest: Obtains information of the busiest areas(i.e. areas of most interest by people) by updating a HeatMap fed by peoples positions.
- Social Distancing: Continuously detect and show possible people at risk for being too close to others by estimating the distance between both.
- Real Time People Counting: Keeps a real time detection of how many people are in the monitored environment(inside camera view).

## Recommendations for using the repository
- Use conda to create virtual environments.
- Use cuda and cudnn for better performance (verify if GPU is compatible).
- Linux is recommended for all procedures.

## Installation

### Python 

Recommended version 3.7.

### Create a conda virtual environment

`conda create -n <environment-name> python=3.7`<br/>
`conda activate <environment-name>` 

### Necessary dependencies

### Download the repository
Download the repository or clone by executing in the shell `git clone https://github.com/FabricioAmoroso/People-Flow-Monitoring.git`. After this steps it will be ready to use.

### Files guide

### Code implementations

## Informations
This project is part of the RAS Unesp Bauru projects. For more information about this and other projects, access: https://sites.google.com/unesp.br/rasunespbauru/home.

## Authors

- [**Artur Starling**](https://github.com/ArturStarling)
- [**Fabrício Amoroso**](https://github.com/FabricioAmoroso)
- [**Gustavo Stahl**](https://github.com/GustavoStah)

## License

This project is free and non-profit.

## Credits

