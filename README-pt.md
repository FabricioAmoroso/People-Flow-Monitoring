# People-Flow-Monitoring

<img src="https://i.imgur.com/gVSsYe8.png" width="250" height="250">

*Disponível em outras línguas: [Inglês](https://github.com/FabricioAmoroso/People-Flow-Monitoring/blob/master/README.md), [Português](https://github.com/FabricioAmoroso/People-Flow-Monitoring/blob/master/README-pt.md).*

> **ATENÇÃO** Esse README não está atualizado com as alterações feitas nessa branch

## Project operation

### Descrição

O seguinte projeto tem o propósito de monitorar ambientes fechados para detectar o fluxo de pessoas.
Existem três características principais: Detecção de área de interesse, Monitoramento de distância social e Contagem de pessoas em tempo real.

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

### Guia de arquivos

### Code implementations

## Informações
This project is part of the RAS Unesp Bauru projects. For more information about this and other projects, access: https://sites.google.com/unesp.br/rasunespbauru/home.

## Autores

- [**Artur Starling**](https://github.com/ArturStarling)
- [**Fabrício Amoroso**](https://github.com/FabricioAmoroso)
- [**Gustavo Stahl**](https://github.com/GustavoStah)
- [**Rafael Conrado**](https://github.com/RafaelRagozoni)

## Licença

Este projeto é gratuito e sem fins lucrativos.

## Créditos


<img src="https://i.imgur.com/mksAQKw.png" width="250" height="250">
