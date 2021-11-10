#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:36:20 2021

@author: dgiron
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



def pinta(x, y, x_ac, y_ac, xlbl, ylbl, xlim, ylim, pos_leyenda, a, lbl, colors):
    """
    Pinta las graficas de los ejercicios

    Parameters
    ----------
    x : list(list)
        valores del eje x iniciales (tiene que ser un array de arrays, aunque solo tenga un elemento).
    y : list(list)
        valores del eje y iniciales (tiene que ser un array de arrays, aunque solo tenga un elemento).
    x_ac : list(list)
        valores del eje x actuales (tiene que ser un array de arrays, aunque solo tenga un elemento)..
    y_ac : list(list)
        valores del eje y actuales (tiene que ser un array de arrays, aunque solo tenga un elemento)..
    xlbl : str
        label del eje x.
    ylbl : str
        label del eje y.
    xlim : tuple
        limites del eje x.
    ylim : tuple
        limites del eje y.
    pos_leyenda : str
        posicion de la segunda leyenda.
    a : bool
        si es True se invierte el eje x.
    lbl : list
        lista con las label para utilizar en la leyenda de masas.
    colors : list
        lista con los colores para utilizar en la leyenda de masas.

    Returns
    -------
    None.

    """
    plt.close()
    plt.clf()
    fig, ax = plt.subplots()
    for i in range(len(x)):
        ax.plot(x[i], y[i], colors[i] + '.')
        ax.plot(x_ac[i], y_ac[i], colors[i] + 'x', alpha=0.5)
        ax.plot([], [], 's'+colors[i][0], label=lbl[i])
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.grid()
    if a:
        fig.gca().invert_xaxis()
    
    leg = ax.legend(title='Masa estrella')
    ax.add_artist(leg)
   
    h = [ax.plot([],[], i+'k')[0] for i in ('-.', '-x')]
    leg3 = ax.legend(handles=h, labels=('t = 0 (creación)', 't = 123 Ma (ahora)'), loc=pos_leyenda, title="Edad estrella")
    ax.add_artist(leg3)
    # plt.savefig('fig4.png', dpi=720)
    fig.show()


   
def hace_array(array_dfs, column):
    """
    Funcion que crea un array con valores de una magnitud para las 6 masas. Por ejemplo,
    un array con las luminosidades de las 6 masas. Sirve para los primeros plots

    Parameters
    ----------
    array_dfs : list
        lista con los datos de las 6 estrellas.
    column : str
        parametro para extraer los datos.

    Returns
    -------
    np.ndarray
        array con el valor de una magnitud 'column' para las 6 masas.

    """
    return np.array([i[column] for i in array_dfs])
  
    
def tabla_latex(tabla, ind, col):
    """
    Prints an array in latex format
    Args:
        tabla: array to print in latex format (formed as an array of the variables (arrays) that want to be displayed)
        ind: list with index names
        col: list with columns names
        r: number of decimals to be rounded
        
    Returns:
        ptabla: table with the data in a pandas.Dataframe format
    """
    tabla = tabla.T
    tabla = np.round(tabla, 2)
    ptabla = pd.DataFrame(tabla, index=ind, columns=col)
    print("Tabla en latex:\n")
    print(ptabla.to_latex(escape=False))
    return ptabla  

def main():
    column_labels = ['step', 'age', 'mass', 'luminosity', 'radius', 'surface_temperature', 'central_temperature',
               'central_density', 'central_pressure', 'psi_c', 'x_c', 'y_c', 'central_carbon_frac',
               'central_nitrogen_frac', 'central_oxigen', 'dynamical_timescale', 'k-h_timescale', 
               'nuclear_timescale', 'luminosity_pp', 'luminosity_cno', 'luminosity_3_alpha',
               'luminosity_metal_burning', 'luminosity_neutrino', 'mass_helium_core', 'mass_carbon_core',
               'mass_oxigen_core', 'radius_helium_core', 'radius_carbon_core', 'radius_oxigen_core']
    
    column_labels_detailed = ["Lagrangian mass coordinate", "Radius coordinate", "Luminosity", "Total pressure",
                               "Density", "Temperature", "Specific internal energy", "Specific entropy", "Specific heat at constant pressure",
                               "First adiabatic exponent", "Adiabatic temperature gradient", "Mean molecular weight", "Electron number density",
                               "Electron pressure", "Radiation pressure", "Radiative temperature gradient", "Material temperature gradient",
                               "Convective velocity", "Rosseland mean opacity", "Power per unit mass from all nuclear reactions",
                               "Power per unit mass from PP chain", "Power per unit mass from CNO cycle", "Power per unit mass from 3-alpha reaction",
                               "Power loss per unit mass in nuclear neutrinos", "Power loss per unit mass in non-nuclear neutrinos",
                               "Power per unit mass from gravitational contraction", "Hydrogen mass fraction", "Molecular hydrogen mass fraction",
                               "Singly-ionized hydrogen mass fraction", "Helium mass fraction", "Singly-ionized helium mass fraction",
                               "Doubly-ionized helium mass fraction", "Carbon mass fraction", "Nitrogren mass fraction",
                               "Oxygen mass fraction", "Electron degeneracy parameter"]
    
    df = []
    data = ['0_1_solar_masses/summary.txt', '0_5_solar_masses/summary.txt', '1_solar_masses/summary.txt',
            '2_solar_masses/summary.txt', '3_solar_masses/summary.txt', '4_5_solar_masses/summary.txt']
    
    # Guarda en una lista los datos para cada masa. Los datos de cada masa estan guardados en un pandas.Dataframe
    for i in range(6):
        a = np.genfromtxt(data[i])
        df.append(pd.DataFrame(a[:, :], columns=column_labels, index=[a[:, 1]]))
    
    # Algoritmo para buscar los datos 123 millones de años despues de que se formase la estrella
    closest_index = []
    step = []
    minimo = math.inf

    for i in df:
        minimo = math.inf
        for j, x in zip(i.index, i['step']):
            if np.abs(j[0] - 123000000) < minimo:
                minimo = np.abs(j[0] - 123000000)
                aux = j[0]
                aux2 = x

        closest_index.append(aux)
        step.append(str(int(aux2)))
    
    # Separa los datos iniciales y actuales del resto, para manipularlos mas facilmente
    iniciales = [i.loc[0, :] for i in df]
    actuales = [i.loc[closest_index[j]] for j, i in enumerate(df)]

    labels = [r'0.1 $({M}_\odot)$', r'0.5 $({M}_\odot)$', r'1 $({M}_\odot)$',
            r'2 $({M}_\odot)$', r'3 $({M}_\odot)$', r'4.5 $({M}_\odot)$']

    colors = ['r', 'b', 'g', 'y', 'm', 'k']

    def ej1():
        pinta(hace_array(iniciales, 'age'), hace_array(iniciales, 'radius'),
              hace_array(actuales, 'age'), hace_array(actuales, 'radius'), r'$\log_{10}{T_s} $(K)', r'$\log_{10}{(\frac{L}{{L}_\odot})}$',
              (3.4, 4.3), (-3.5, 3.0), 'lower left', True, labels, colors)
    def ej2():
        pinta(hace_array(iniciales, 'central_temperature'), np.log10(hace_array(iniciales, 'y_c')/hace_array(iniciales, 'x_c')),
              hace_array(actuales, 'central_temperature'), np.log10(hace_array(actuales, 'y_c')/hace_array(actuales, 'x_c')), 
              r'$\log_{10}{T_c}$(K)', r'$\log_{10}{(\frac{Y_c}{X_c})}$', (6.4, 7.7), (-0.5, 1.3), 'upper center', True, labels, colors)
   
    
    # Guarda datos detallados para la estrella mas masiva en el momento de su formacion y en la actualidad
    detailed_data = np.genfromtxt('4_5_solar_masses/structure_00000.txt')
    df_detailed = pd.DataFrame(detailed_data, columns=column_labels_detailed)
    
    detailed_data_m_4_5 = np.genfromtxt('4_5_solar_masses/structure_000{}.txt'.format(step[-1]))
    df_detailed_actual = pd.DataFrame(detailed_data_m_4_5, columns=column_labels_detailed)

    def ej3():
        pinta([df_detailed['Radius coordinate']], [np.log10(df_detailed['Radiation pressure'])], 
              [df_detailed_actual['Radius coordinate']], [np.log10(df_detailed_actual['Radiation pressure'])],
              r'$\frac{r}{{R}_\odot}$', r'$\log_{10}{(P_{rad}}$ (CGS))', (1.5, 6.5), (0, 13), 'lower center', False, [labels[-1]], [colors[-1]+'-'])
       

    def ej4():
        pinta([df_detailed['Radius coordinate']], [np.log10(df_detailed['Power per unit mass from all nuclear reactions'])], 
              [df_detailed_actual['Radius coordinate']], [np.log10(df_detailed_actual['Power per unit mass from all nuclear reactions'])],
              r'$\frac{r}{{R}_\odot}$', r'$\log_{10}{(\epsilon_{nuc}}$(CGS))', (0, 0.2), (3, 4.5), 'lower center', False, [labels[-1]], [colors[-1]+'-'])


    # Tabla con las luminosidades para las dos masas mayores
    lumin_pp = [i['luminosity_pp'].iloc[0] for i in actuales[-2:]]
    lumin_cno = [i['luminosity_cno'].iloc[0] for i in actuales[-2:]]
    lumin_3a = [i['luminosity_3_alpha'].iloc[0] for i in actuales[-2:]]
    t = [10**i['central_temperature'].iloc[0] for i in actuales[-2:]]

    tabla = np.array([lumin_pp, lumin_cno, lumin_3a, t])
    print(tabla_latex(tabla, ind=['3 M$_\odot$', '4.5 M$_\odot$'], col=['$L_{pp}$', '$L_{CNO}$', '$L_{3\alpha}$', 'T']))
    
    # Descomentar para pintar las graficas
    #ej1()
    #ej2()
    #ej3()
    #ej4()

main()