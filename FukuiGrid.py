import numpy as np
from numpy.linalg import norm
from scipy import stats, ndimage
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
import os
from scipy.interpolate import interp1d

############################################################
#                   Utility Functions                      #
############################################################

def read_file_info(FILE1):
    NATOMS=0 
    GRID, POINTS = None, None
    alattvec, blattvec, clattvec = None, None, None
    
    with open(FILE1) as fp:
        for i, line in enumerate(fp):
            if i == 2:
                alattvec = list(map(float, line.split()))
            elif i == 3:
                blattvec = list(map(float, line.split()))
            elif i == 4:
                clattvec = list(map(float, line.split()))
            elif i == 6:
                line = map(int, line.split())
                NATOMS = sum(line)
            elif i == NATOMS + 9:
                GRID = list(map(int, line.split()))
                POINTS = np.prod(GRID)
            elif i > NATOMS + 11:
                break
    skiplines = NATOMS + 9
    if POINTS % 5 == 0:
        MAXROWS = int(POINTS/5)
    elif POINTS % 5 != 0:
        MAXROWS = int(np.ceil(POINTS/5))
    print('NATOMS = ', NATOMS, 'GRID = ',GRID, 'POINTS = ',POINTS)
    return NATOMS, GRID, POINTS, alattvec, blattvec, clattvec, skiplines, MAXROWS

def process_file(filename, skip_lines, max_rows, points):
    df = pd.read_fwf(filename, colspecs='infer', skiprows=skip_lines, nrows=max_rows)
    data = df.to_numpy().flatten('C')[:points]
    return data

def calculate_fukui_and_r2(dn, CHG1, CHG2, CHG3, CHG4):
    density_matrix = np.vstack([CHG1, CHG2, CHG3, CHG4]).astype(float)
    coeffs = np.polyfit(dn, density_matrix, deg=1) 
    slopes = coeffs[0]  
    correlation_matrix = np.corrcoef(density_matrix)
    r_squared_values = correlation_matrix[0, 1] ** 2

    return slopes, r_squared_values

def write_fukui_file(FILE1, NATOMS, fukui, filename):
    ### Abrir el archivo de salida para escribir todo de una vez
    with open(filename, "w") as FUKUIFILE:
        
        # Escribir el encabezado desde FILE1
        with open(FILE1) as fp:
            for i, line in enumerate(fp):
                if i < NATOMS + 10:
                    FUKUIFILE.write(line)
                else:
                    break

        # Ahora procesamos y escribimos los datos de 'fukui'
        num_full_rows = fukui.size // 5
        last_row_size = fukui.size % 5
        
        if last_row_size == 0:
            fukui_matrix = fukui.reshape(-1, 5)
        else:
            fukui_matrix = np.zeros((num_full_rows + 1, 5))
            fukui_matrix[:num_full_rows] = fukui[:num_full_rows * 5].reshape(-1, 5)
            fukui_matrix[-1, :last_row_size] = fukui[num_full_rows * 5:]
        
        # Escribir los datos de 'fukui' en el archivo
        for row in fukui_matrix:
            formatted_row = " ".join(f"{value: .11E}" for value in row if value != 0)
            FUKUIFILE.write(formatted_row + "\n")
    print('file saved: ',filename)
    return filename

def reshape_xyz(NGX,NGY,NGZ, CHG):
    GHGtem =  np.zeros((NGX,NGY,NGZ))
    for N3 in range(NGX):
        for N2 in range(NGY):
            for N1 in range(NGZ):
                GHGtem[N3,N2,N1] = CHG[N1*NGX*NGY + N2*NGX + N3]
    return GHGtem

def missing(data, POINTS):
    nan_count=data.isna().sum().sum()  
    if nan_count==0:
        CHG=data.to_numpy()
        CHG=CHG.flatten('C')
        CHG=CHG[0:POINTS]
        return CHG

    else:
        CHG=data.to_numpy()
        CHG=CHG.flatten('C')
        CHG=CHG[0:POINTS]
        hay_nan = any(np.isnan(valor) for valor in CHG)
        if hay_nan:
            print("Input error")
        else:
            print("")
        return CHG

def func_correction(omega_vol, eps, LS, LZ, NGZ):
        q = 1.6022 *1e-19
        eps_0 = 8.854*1e-22 
        modulo = (-q/(2.0*eps_0*omega_vol)) 
        correction = []
        recorrido = np.linspace(-(LZ*0.5),(LZ*0.5) , NGZ)
        for z in recorrido:
            if (-LS*0.5) < z < (LS*0.5):
                func_dentro = (1/eps)*(z*z + LS*LS*0.25*(eps -1) - eps*LZ*LZ*0.25)
                correction.append(func_dentro*modulo)
            elif (abs(z)>= (LS*0.5)):
                func_fuera = z*z - (LZ*LZ*0.25)
                correction.append(func_fuera*modulo)
            else:
                print("Error func_correction") ### aqui hubo una i
        return correction

def compute_lattice_parameters(alattvec, blattvec, clattvec):
    LATTCURA = (1.0 / 0.529177210903) * np.dstack([alattvec, blattvec, clattvec])
    LATTCURA = LATTCURA[0]
    LATTCURB = np.zeros((3, 3))
    LATTCURB[0] = np.cross(LATTCURA[1], LATTCURA[2])
    LATTCURB[1] = np.cross(LATTCURA[2], LATTCURA[0])
    LATTCURB[2] = np.cross(LATTCURA[0], LATTCURA[1])
    VOL = np.abs(np.linalg.det(np.dstack([alattvec, blattvec, clattvec])))
    omega = np.dot(np.cross(LATTCURA[1], LATTCURA[2]), LATTCURA[0])
    LATTCURB /= omega
    return LATTCURB, VOL, omega

def compute_gsquared(GRID, LATTCURB, q):
    NGX, NGY, NGZ = GRID
    LPCTX = np.array([NX if NX < int(NGX / 2) + 1 else NX - NGX for NX in range(NGX)])
    LPCTY = np.array([NY if NY < int(NGY / 2) + 1 else NY - NGY for NY in range(NGY)])
    LPCTZ = np.array([NZ if NZ < int(NGZ / 2) + 1 else NZ - NGZ for NZ in range(NGZ)])
    GSQU = np.zeros((NGX, NGY, NGZ))
    
    for N3 in range(NGZ):
        for N2 in range(NGY):
            for N1 in range(NGX):
                GX = LPCTX[N1] * LATTCURB[0, 0] + LPCTY[N2] * LATTCURB[0, 1] + LPCTZ[N3] * LATTCURB[0, 2]
                GY = LPCTX[N1] * LATTCURB[1, 0] + LPCTY[N2] * LATTCURB[1, 1] + LPCTZ[N3] * LATTCURB[1, 2]
                GZ = LPCTX[N1] * LATTCURB[2, 0] + LPCTY[N2] * LATTCURB[2, 1] + LPCTZ[N3] * LATTCURB[2, 2]
                GSQU[N1, N2, N3] = (GX * GX + GY * GY + GZ * GZ)
    
    GSQU[0, 0, 0] = 1.0
    GSQU = 1.0 / (GSQU * (2.0 * np.pi * 2.0 * np.pi))
    GSQU[0, 0, 0] = q
    return GSQU

def compute_planar_average_nz(CHG00, NGX, NGY, NGZ, axis):
    if axis == 'x':
        return [np.sum(CHG00[nx, :, :]) / (NGY * NGZ) for nx in range(NGX)]
    if axis == 'y':
        return [np.sum(CHG00[:, ny, :]) / (NGX * NGZ) for ny in range(NGY)]
    if axis == 'z':
        return [np.sum(CHG00[:, :, nz]) / (NGY * NGX) for nz in range(NGZ)]

def plot_planar_average(z_axis, PLANAR1, PLANAR3):
    plt.plot(z_axis, PLANAR3, label='Corrected', linewidth=2)
    plt.plot(z_axis, PLANAR1, label='No correction', linewidth=2)
    plt.title('Planar Average Fukui Potential', fontsize=18)
    plt.ylabel(r'$v_{f}(r)$ (eV)', fontsize=12.5)
    plt.xlabel(r'Z-direction ($\AA$)', fontsize=12.5)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend()
    plt.savefig('PA_vFuk.svg')

def read_data(FILE0, skiplines, maxrows, POINTS):
    if (POINTS % 5) != 0:
      with open(FILE0, 'r') as file:
         last_line = file.readlines()[maxrows + skiplines]
         df0 = pd.read_table(FILE0, sep=r'\s+', skiprows=skiplines+1, names=range(5), nrows=maxrows)
    else:
        skiprowsn=skiplines + 1
        df0 = pd.read_table(FILE0, sep=r'\s+', skiprows=skiplines+1, names=range(5), nrows=maxrows)
    return df0

def closest_value_position(lst, number, percentage=0.4):
    num_elements = int(len(lst) * percentage)
    position, closest_value = min(enumerate(lst[-num_elements:]), key=lambda x: abs(x[1] - number))
    return closest_value, position + len(lst) - num_elements

def extract_data_from_file(file_path, marker='#z-vcor.dat'):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the start of the data after the marker
    data_start = None
    for i in range(len(lines)-1, -1, -1):
        if lines[i].strip().startswith(marker):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"The file does not contain the marker '{marker}'")

    # Extract the data starting from the marker
    data_lines = lines[data_start:]
    z2, value2 = [], []
    for line in data_lines:
        if line.strip():
            cols = line.split()
            z2.append(float(cols[0]))
            value2.append(float(cols[1]))

    # Convert to numpy arrays
    return np.array(z2), np.array(value2)

def write_xyz(CHG, NGX, NGY, NGZ, LATTCURADO, omega, output_file='xyz_value.dat'):
    suma = 0.0
    with open(output_file, 'w') as xyz_file:
        # Write the header for the XYZ file
        xyz_file.write(f"{NGX * NGY * NGZ}\n")
        xyz_file.write("Generated XYZ file\n")

        # Iterate over the grid and calculate coordinates and charge density
        for N3 in range(NGZ):
            for N2 in range(NGY):
                for N1 in range(NGX):
                    # Calculate the X, Y, Z coordinates
                    x = (N1 * LATTCURADO[0, 0] + N2 * LATTCURADO[1, 0] + N3 * LATTCURADO[2, 0]) / NGX
                    y = (N1 * LATTCURADO[0, 1] + N2 * LATTCURADO[1, 1] + N3 * LATTCURADO[2, 1]) / NGY
                    z = (N1 * LATTCURADO[0, 2] + N2 * LATTCURADO[1, 2] + N3 * LATTCURADO[2, 2]) / NGZ

                    # Get the charge density value at the current grid point
                    valor_densidad = CHG[N1, N2, N3]
                    
                    # Accumulate the sum of charge density values
                    suma += valor_densidad

                    # Write the coordinates and charge density to the file
                    xyz_file.write(f"{x:>12.4f}{y:>12.4f}{z:>12.4f}{valor_densidad:>20.8e}\n")

    # Calculate the differential element
    Elemento_volumen = omega / (NGX * NGY * NGZ)
    
    # Calculate the integral of the charge density
    Integral_rho = suma * Elemento_volumen
    return output_file,Integral_rho

def detect_local_extrema_3D(data, order, extrema_type='min'):
    """
    Detects local extrema (maxima or minima) in a 3D array.

    Parameters
    ----------
    data : ndarray
        3D array of data.
    order : int
        Number of points on each side to use for comparison.
    extrema_type : str
        Type of extrema to detect ('min' for minima, 'max' for maxima).

    Returns
    -------
    coords : ndarray
        Coordinates of the local extrema.
    values : ndarray
        Values of the local extrema.
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    if extrema_type == 'min':
        # Detect local minima
        filtered = ndimage.minimum_filter(data, footprint=footprint, mode='wrap')
        mask_extrema = data < filtered
    elif extrema_type == 'max':
        # Detect local maxima
        filtered = ndimage.maximum_filter(data, footprint=footprint, mode='wrap')
        mask_extrema = data > filtered
    else:
        raise ValueError("extrema_type must be 'min' or 'max'")

    coords = np.asarray(np.where(mask_extrema)).T
    values = data[mask_extrema]

    return coords, values

def write_formatted_data_localm(filename, data):
    with open(filename, 'w') as archivo:
        for fila in data:
            archivo.write(f"{fila[0]:>10.7f} {fila[1]:>10.7f} {fila[2]:>10.7f} {fila[3]:>10.7f}\n")
    print(f"Data successfully written to {filename}")

def calcular_xyz_val(coords, values, GRID, alattvec, blattvec, clattvec):
    
    min_xyz_vals = []

    for i in range(len(values)):
        NZ = coords[i][0]
        NY = coords[i][1]
        NX = coords[i][2]

        x_min = ((NX - 1) / GRID[0]) * np.linalg.norm(np.array(alattvec), 2)
        y_min = ((NY - 1) / GRID[1]) * np.linalg.norm(np.array(blattvec), 2)
        z_min = ((NZ - 1) / GRID[2]) * np.linalg.norm(np.array(clattvec), 2)

        min_xyz_val = np.array([x_min, y_min, z_min, values[i]])

        min_xyz_vals.append(min_xyz_val)

    return min_xyz_vals

def convert_locpot(LOCPOT_cor2, NGX, NGY, NGZ):
    LOCPOTtem = np.zeros(NGX * NGY * NGZ)
    for N1 in range(NGZ):
        for N2 in range(NGY):
            for N3 in range(NGX):
                LOCPOTtem[N1 * NGX * NGY + N2 * NGX + N3] = - LOCPOT_cor2[N3, N2, N1]
                #cambiamos el signo al tiro
    return LOCPOTtem

def read_xyzval(archivo):
    datos = []
    with open(archivo, 'r') as f:
        for linea in f:
            try:
                x, y, z, valor = map(float, linea.strip().split()[:4])
                datos.append((x, y, z, valor))
            except ValueError:
                continue
    return np.array(datos)

def compare_columns(col1, col2, tolerancia=1e-8):
    #Compare xyz columns between two files
    if col1.shape != col2.shape:
        print("The files have different sizes.")
        return False
    if np.allclose(col1[:, :3], col2[:, :3], atol=tolerancia):
        print("The columns are the same between the files.")
        return True
    else:
        diferencias = np.where(~np.isclose(col1[:, :3], col2[:, :3], atol=tolerancia))
        print("Differences found in lines:", diferencias[0] + 1)
        return False
    
def filter_values(datos1, datos2, ztol):
    """Filtering data with density close to 1e-3."""
    # Filter value
    #filtro = (0.0009 <= datos1[:, 3]) & (datos1[:, 3] <= 0.0011) & (datos1[:, 2] >= ztol)
    filtro = (0.0001 <= datos1[:, 3]) & (datos1[:, 3] <= 0.0003) & (datos1[:, 2] >= ztol)
    datos_filtrados = datos1[filtro]

    # choosing data closer to 1e-3
    xy_coords = {}
    for x, y, z, valor in datos_filtrados:
        clave = (x, y)
        if clave not in xy_coords or abs(valor - 2e-4) < abs(xy_coords[clave][1] - 2e-4):
        #if clave not in xy_coords or abs(valor - 1e-3) < abs(xy_coords[clave][1] - 1e-3):
            xy_coords[clave] = (z, valor)

    # choosing values in MODELPOT file
    coordenadas_valores_archivo2 = {(x, y, z): valor for x, y, z, valor in datos2}
    resultado = [(x, y, z, coordenadas_valores_archivo2.get((x, y, z), None))
                 for (x, y), (z, valor) in xy_coords.items() if (x, y, z) in coordenadas_valores_archivo2]

    print(f"Number of data filtered in MODELPOT file: {len(resultado)}")
    return resultado

############################################################
#                     Main Functions                       #
############################################################

def Fukui_interpolation(FILE1,FILE2,FILE3,FILE4, dn=None):    
    """
    Computes the Fukui function via interpolation using charge density data from multiple electronic configurations.

    Parameters:
    -----------
    FILE1 : str
        Path to the first charge density file.
    FILE2 : str
        Path to the second charge density file.
    FILE3 : str
        Path to the third charge density file.
    FILE4 : str
        Path to the fourth charge density file.
        These files must be provided in the same order as the electron perturbations in the `dn` array.
    dn : numpy.ndarray, optional
        Array of electron perturbation values corresponding to the charge density files.
        Default: `np.array([-0.15, -0.1, -0.05, 0.0])`. 
        Represents deviations from the neutral configuration (e.g., ±0.15 electrons).

    Returns:
    --------
    final_file : str
        Path to the output file containing the interpolated Fukui function in VASP format.

    Description:
    ------------
    This function processes charge density data from four input files, each corresponding to a system with 
    a different number of electrons. It calculates the Fukui function using interpolation based on the 
    provided `dn` values, representing each configuration's electron perturbations.

    Notes:
    ------
    - Ensure the `dn` array is consistent with the charge densities.
    - This function relies on auxiliary functions such as `read_file_info`, `process_file`, and `calculate_fukui_and_r2`.

    Example Usage:
    --------------
    ```
    dn = np.array([-0.15, -0.1, -0.05, 0.0])
    final_file = Fukui_interpolation("CHGCAR1", "CHGCAR2", "CHGCAR3", "CHGCAR_neutral", dn=dn)
    print("Output written to:", final_file)
    ```
    """

    if dn is None:
        dn = np.array([-0.15, -0.1, -0.05, 0.0])
    print(r"\delta N is: ", dn)
    print ("Reading info from files")

    #  Reads structural and grid information from `FILE1` and processes the charge density data from `FILE1`, `FILE2`, `FILE3`, and `FILE4`.
    
    NATOMS, GRID, POINTS, alattvec, blattvec, clattvec, skiplines, maxrows = read_file_info(FILE1)
    
    print("Collecting info from different files")
    CHG1 = process_file(FILE1, skiplines, maxrows, POINTS)
    CHG2 = process_file(FILE2, skiplines, maxrows, POINTS)
    CHG3 = process_file(FILE3, skiplines, maxrows, POINTS)
    CHG4 = process_file(FILE4, skiplines, maxrows, POINTS)
    
    print('CHG1',CHG1)
    print('CHG2',CHG2)
    print('CHG3',CHG3)
    print('CHG4',CHG4)

    # Performs the interpolation to compute the Fukui function and R-squared values.
    print('Performing interpolation')

    FUKUI, RSQUARED = calculate_fukui_and_r2(dn, CHG1, CHG2, CHG3, CHG4)

    print('Fukui size', FUKUI.size)
    print('Fukui', FUKUI) 
    # Writes the resulting Fukui function to a new file, `CHGCAR_FUKUI.vasp`.
    final_file= write_fukui_file(FILE1, NATOMS, FUKUI, "CHGCAR_FUKUI.vasp")
    
    print('FUKUI file was written', final_file)
    print("")
    return final_file

def fukui_electrodes(FILE0,FILE1,Epsilon):
    """
        Applies corrections to the electrostatic potential to calculate the Fukui function for electrode systems.

        Parameters:
        -----------
        FILE0 : str
            Path to the charge density file of the neutral slab.
        FILE1 : str
            Path to the Fukui function file.
        Epsilon : float
            Dielectric constant for the material under study.

        Returns:
        --------
        final_file : str
            Path to the output file containing the Fukui Potential in VASP format.
        Additionally, it plots the planar averages of the potential to visualize the corrections.

        Description:
        ------------
        This function processes charge density and electrostatic potential data from electrode systems. 
        It applies corrections for periodic boundary conditions, computes planar averages, and adjusts the electrostatic 
        potential for the system under study. The final corrected electrostatic potential is used to compute the Fukui LOCPOT.
    """
    NATOMS, GRID, POINTS, alattvec, blattvec, clattvec, skiplines, maxrows = read_file_info(FILE1)
    
    NGX,NGY,NGZ = GRID
    
    print("This will take a few seconds.")
    print("")

    LATTCURB, VOL, omega= compute_lattice_parameters(alattvec, blattvec, clattvec)
    GSQU=compute_gsquared(GRID, LATTCURB, 0)        

    df0 = read_data(FILE0,skiplines, maxrows, POINTS)
    df = read_data(FILE1,skiplines, maxrows, POINTS)
                    
    CHG = missing(df,POINTS)
    CHG00 = missing(df0,POINTS)

    CHGtem = reshape_xyz(NGX, NGY, NGZ, CHG)
    CHGtem00 = reshape_xyz(NGX, NGY, NGZ, CHG00)

    CHG = CHGtem/omega
    CHG00 = CHGtem00/omega

    #Corrections and Poisson equations for the backgroun

    CHGG = np.fft.fftn(CHG, norm='ortho')    
    LOCPOTG = 4*np.pi*np.multiply(CHGG,GSQU)
    LOCPOT = np.fft.ifftn(LOCPOTG,norm='ortho').real * 27.2114
    
    #align the background 1e-4
    cota = 0.0001
    cota_ls = 0.02

    PLANAR_CHG00 = compute_planar_average_nz(CHG00, NGX, NGY, NGZ, 'z')
    valor_cercano, pos0 = closest_value_position(PLANAR_CHG00, cota)
    val_ls, pos_ls = closest_value_position(PLANAR_CHG00, cota_ls)
    
    LZ = clattvec[2] 
    LS = ((LZ/NGZ)*pos_ls - LZ/2)*2
    
    salida = [item for sublist in func_correction(VOL, Epsilon, LS, LZ, NGZ) for item in np.array(sublist).flatten() ]
    salida = np.array(salida)
    LOCPOT_cor = LOCPOT + salida[np.newaxis, np.newaxis, :]
        
    # info for the plot
    PLANAR1 = compute_planar_average_nz(LOCPOT, NGX, NGY, NGZ, 'z')
    PLANAR2 = compute_planar_average_nz(LOCPOT_cor, NGX, NGY, NGZ, 'z')
        
    cota2 = PLANAR2[pos0]
    PLANAR2f_2 = [elemento - cota2 if 0 <= (elemento - cota2) else 0 for elemento in PLANAR2] 
    correccion_final = [a - b for a, b in zip(PLANAR2f_2, PLANAR2)]
    correccion_final = np.array(correccion_final)
    LOCPOT_cor2 = LOCPOT_cor + correccion_final[np.newaxis, np.newaxis, :]

    PLANAR3= compute_planar_average_nz(LOCPOT_cor2, NGX, NGY, NGZ, 'z')
      
    PLANAR1 = [-x for x in PLANAR1]
    PLANAR3 = [-x for x in PLANAR3]

    z_axis = np.linspace(0, LZ, NGZ)
    
    plot_planar_average(z_axis, PLANAR1, PLANAR3)

    # Write the file
    LOCPOTtem = convert_locpot(LOCPOT_cor2, NGX, NGY, NGZ)

    final_file = write_fukui_file(FILE1, NATOMS, LOCPOTtem, 'FUKUI.LOCPOT')

    return final_file
  
def fukui_SCPC(FILE0,FILE1,FILE2,c):
    """
    Computes the Fukui Potential using self-consistent potential corrections (SCPC) and outputs the corrected data.

    Parameters:
    -----------
    FILE0 : str
        Path to the reference charge density file (neutral slab).
    FILE1 : str
        Path to the input charge density file of Fukui function.
    FILE2 : str
        Path to the file containing correction values as a function of the z-axis (e.g., '#z-vcor.dat').
    c : 1 or -1
        Scaling factor for applying corrections based on the external potential. 
        c = -1 for Electrophilic Fukui function v_f^-(r)
        c = 1  for Nucleophilic Fukui function v_f^+(r).

    Returns:
    --------
    final_file : str
        Path to the output file ('FUKUI.LOCPOT') containing the corrected Fukui Potential 
        in the format compatible with VASP.

    Description:
    ------------
    This function calculates the Fukui function by processing charge density data and applying 
    self-consistent corrections to the electrostatic potential. 

    Notes:
    ------
    - The input files must have consistent grid sizes and formats.
    - The correction values from `FILE2` are interpolated using cubic interpolation to match the grid of `FILE1`.
    - The output file is written in a format suitable for further analysis in VASP.

    Example:
    --------
    ```
    final_file = fukui_SCPC('CHGCAR_neutral', 'CHGCAR_perturbed', 'z-vcor.dat', c=1)
    print(f"Corrected file for for Nucleophilic Fukui function saved to {final_file}")
    ```
    """

    # Reading charge density data from `FILE0` (neutral) and `FILE1` (Fukui)
    NATOMS, GRID, POINTS, alattvec, blattvec, clattvec, skiplines, maxrows = read_file_info(FILE1)
    
    NGX,NGY,NGZ = GRID
    
    print("This will take a few seconds.")
    print("")
    # Computing the lattice parameters and the reciprocal grid.
    LATTCURB, VOL, omega= compute_lattice_parameters(alattvec, blattvec, clattvec)
    GSQU = compute_gsquared(GRID, LATTCURB, 0)
    
    df0 = read_data(FILE0, skiplines, maxrows, POINTS)
    df = read_data(FILE1, skiplines, maxrows, POINTS)
    
    CHG = missing(df,POINTS)
    CHG00 = missing(df0,POINTS)
    CHGtem = reshape_xyz(NGX, NGY, NGZ, CHG)
    CHGtem00 = reshape_xyz(NGX, NGY, NGZ, CHG00)
    CHG = CHGtem/omega
    CHG00 = CHGtem00/omega

    #Applying Fourier transforms to compute the electrostatic potential (`LOCPOT`) and introducing corrections from the data in `FILE2`.
    PLANAR_CHG00 = compute_planar_average_nz(CHG00, NGX, NGY, NGZ, 'z')
    
    cota = 0.0001
    valor_cercano, pos0 = closest_value_position(PLANAR_CHG00, cota)
    
    CHGG = np.fft.fftn(CHG, norm='ortho')    
    LOCPOTG = 4*np.pi*np.multiply(CHGG,GSQU)
    LOCPOT = np.fft.ifftn(LOCPOTG,norm='ortho')    
    LOCPOT = 27.2114*LOCPOT.real 
    
    LZ = clattvec[2] 
    z_axis = np.linspace(0, LZ, NGZ)

    PLANAR1 = compute_planar_average_nz(LOCPOT, NGX, NGY, NGZ, 'z')
    
    z2, value2 = extract_data_from_file(FILE2, marker='#z-vcor.dat')

    # Interpolating the corrections along the z-axis depend on type of Fukui (c)
    interp2 = interp1d(z2, value2, kind='cubic', fill_value="extrapolate")
    value2_interp = interp2(z_axis)
    
    print('value2_interp=',value2_interp)

    LOCPOT_cor = LOCPOT.copy()
    for i in range(NGZ):
        LOCPOT_cor[:,:,i] = LOCPOT_cor[:,:,i] - value2_interp[i]*(c)

    # Generating corrected potential data and writing the results to an output file.
    PLANAR2 = compute_planar_average_nz(LOCPOT_cor, NGX, NGY, NGZ, 'z')
    cota2 = PLANAR2[pos0]
    PLANAR2f_2 = [elemento - cota2 if 0 <= (elemento - cota2) else 0 for elemento in PLANAR2]
    correccion_final = [a - b for a, b in zip(PLANAR2f_2, PLANAR2)]

    
    LOCPOT_cor2 = LOCPOT_cor.copy()
    for i in range(NGZ):
                LOCPOT_cor2[:,:,i] = LOCPOT_cor2[:,:,i] + correccion_final[i]

    PLANAR3= compute_planar_average_nz(LOCPOT_cor2, NGX, NGY, NGZ, 'z')

    PLANAR1 = [-x for x in PLANAR1]
    PLANAR3 = [-x for x in PLANAR3]
    value2_interp = [-x for x in value2_interp]

    z_axis = np.linspace(0, LZ, NGZ)

    plot_planar_average(z_axis, PLANAR1, PLANAR3)

    LOCPOTtem = convert_locpot(LOCPOT_cor2, NGX, NGY, NGZ)

    final_file = write_fukui_file(FILE1, NATOMS, LOCPOTtem, 'FUKUI.LOCPOT')
    return final_file
    
def lineal_operation(FILE1,FILE2,c1,c2,c3):
    """
    Performs a linear combination of LOCPOT or CHGCAR data from two input files.

    Parameters:
    -----------
    FILE1 : str
        Path to the first charge density file (e.g., CHGCAR or LOCPOT).
    FILE2 : str
        Path to the second charge density file (e.g., CHGCAR or LOCPOT).
    c1 : float
        Coefficient for scaling the charge density from `FILE1`.
    c2 : float
        Coefficient for scaling the charge density from `FILE2`.
    c3 : float
        Constant term to add to the linear combination.

    Returns:
    --------
    final_file : str
        Path to the output file ('CHGCARSUM') containing the resulting charge density
        in VASP-compatible format.

    Notes:
    ------
    - The input files must be formatted consistently and correspond to the same system.
    - The output file is saved in a format compatible with further analysis using VASP.

    Example:
    --------
    ```
    final_file = lineal_operation('CHGCAR1', 'CHGCAR2', 1.5, -0.8, 0.2)
    print(f"Linear combination saved to {final_file}")
    ```
    """

    # Reading data (number of atoms, grid points, and lattice vectors) from `FILE1` and `FILE2`.
    print ("FILE1: ",FILE1)
    NATOMS, GRID, POINTS, alattvec, blattvec, clattvec, skiplines, maxrows = read_file_info(FILE1)
    
    print ("FILE2: ",FILE2)
    NATOMS2, GRID2, POINTS2, alattvec2, blattvec2, clattvec2, skiplines2, MAXROWS2 = read_file_info(FILE2)
    
    # Ensuring that the two files correspond to the same system (i.e., have the same grid size and number of points).
    if POINTS != POINTS2:
        print("The files have different systems")
        return

    # Processing charge density data from both files.     
    CHG1 = process_file(FILE1, skiplines, maxrows, POINTS)
    print('CHG1',CHG1)

    CHG2 = process_file(FILE2, skiplines2, MAXROWS2, POINTS2)
    print('CHG2',CHG2)
    
    # Applying the linear operation and writing the resulting charge density data to an output file (`CHGCARSUM`).
    CHGSUM = c1 * CHG1 + c2 * CHG2 + c3
    
    final_file = write_fukui_file(FILE1, NATOMS, CHGSUM, 'CHGCARSUM')
    return final_file
    
def planar_average(FILE1, type_file, axis='z'):
    """
    Computes the planar average of charge density or electrostatic potential along a specified axis.

    Parameters:
    -----------
    FILE1 : str
        Path to the charge density or electrostatic potential file (e.g., CHGCAR or LOCPOT format).
    type_file : str
        Type of input file ('CHGCAR' or 'LOCPOT') indicating whether the file contains charge density or potential data.
    axis : str, optional, default 'z'
        Axis along which the planar average is computed. Can be 'x', 'y', or 'z'.

    Returns:
    --------
    data_to_save : numpy.ndarray
        Array containing the planar average data, with the first column as the axis values and the second column as the averaged data.
    TITLE_dat : str
        Filename where the planar average data is saved.

    Description:
    ------------
    This function reads a charge density or electrostatic potential file and computes the planar average along a given axis. 
    The result is saved to a `.dat` file and a plot is generated. The function assumes that the input grid is orthogonal.

    Notes:
    ------
    - This function only works for orthogonal lattice vectors (rectangular grids).
    - The file should be formatted as either 'CHGCAR' (charge density) or 'LOCPOT' (electrostatic potential).
    - The axis parameter determines the direction in which the planar average is computed.

    Example:
    --------
    ```
    data, filename = planar_average('CHGCAR', 'CHGCAR', axis='z')
    print(f"Data saved in {filename}")
    ```
    """

    print ("FILE1: ",FILE1)
    NATOMS, GRID, POINTS, alattvec, blattvec, clattvec, skiplines, maxrows = read_file_info(FILE1)
    NGX,NGY,NGZ = GRID
    print(alattvec,blattvec,clattvec)
    LATTCURB, VOL, omega= compute_lattice_parameters(alattvec, blattvec, clattvec)
    LZ = clattvec[2]
    LY = blattvec[1]
    LX = alattvec[0]

    df = read_data(FILE1, skiplines, maxrows, POINTS)
    CHG = df.to_numpy().flatten('C')[0:POINTS]
    GHGtem =  reshape_xyz(NGX, NGY, NGZ, CHG)


    if type_file == 'CHGCAR':
        CHG = GHGtem/omega
        TITLE_plot = 'Planar Average Electron Density'
        TITLE_dat = 'PA_ED.dat'
        y_label = r'$\rho(r) \ (a_{0}^{-3})$'
        name_fig = 'PA_ED.png'
    
    if type_file == 'LOCPOT':
        CHG = GHGtem
        TITLE_plot = 'Planar Average Electrostatic Potential'
        TITLE_dat = 'PA_EP.dat'
        y_label= r'V(r) (eV)'
        name_fig = 'PA_EP.png'

    if axis == 'z':
        xlabel = r'Z-direction ($\AA$)'
        n_axis = np.linspace(0, LZ, NGZ)
    if axis == 'y':
        xlabel = r'Y-direction ($\AA$)'
        n_axis = np.linspace(0, LY, NGY)
    if axis == 'x':
        xlabel = r'x-direction ($\AA$)'
        n_axis = np.linspace(0, LX, NGX)
    
    print('Preparing ',axis, ' axis data')
    print(NGX,NGY,NGZ)
    print(n_axis)
    print(np.linspace(0, LX, NGX))

    AVEGPOTZcorr = compute_planar_average_nz(CHG, NGX, NGY, NGZ, axis)
    data_to_save = np.column_stack((n_axis, AVEGPOTZcorr))
    

    fmt = '%20.6f %20.5e'
    header = f'{"n_axis".rjust(20)}{"Planar_Avg".rjust(20)}'

    np.savetxt(TITLE_dat, data_to_save, fmt=fmt, header=header, comments='')    
    plt.clf()
    plt.plot(n_axis,AVEGPOTZcorr,linewidth=2)
    plt.title(TITLE_plot,fontsize=18)
    plt.ylabel(y_label,fontsize=12.5)
    plt.xlabel(xlabel,fontsize=12.5)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()
    plt.savefig(name_fig)
    return data_to_save, TITLE_dat

def XYZ_value(FILE1,type_file, c1=1,align_pot=False,plot=False):
    """
    Converts charge density or electrostatic potential data from VASP files to XYZ format, and optionally 
    align potential to zero in the vacuum, change to the chemical convention (i.e. take the negative) 
    and generates a plot for planar average of LOCPOT on z-axis.

    Parameters:
    -----------
    FILE1 : str
        Path to the input file (e.g., CHGCAR or LOCPOT format) containing charge density or electrostatic potential data.
    type_file : str
        Type of the input file ('CHGCAR' for charge density or 'LOCPOT' for electrostatic potential).
    c1 : 1 or -1, default 1 (used only when `type_file` is 'LOCPOT').
        1  Physic convertion
        -1 Chemical convertion
    align_pot : bool, optional, default False
        If True, align potential to zero in the vacuum (used only when `type_file` is 'LOCPOT').
    plot : bool, optional, default False
        If True, generates a plot of the planar average of the electrostatic potential (only for 'LOCPOT' files).

    Returns:
    --------
    final_file : str
        The path to the output XYZ file containing the processed data.
    

    Notes:
    ------
    - For the plot, this function assumes that the input grid is orthogonal and that the data is in the correct format.
    - The output XYZ file will have the name format "XYZ_{type_file}.dat" where `type_file` is 'CHGCAR' or 'LOCPOT'.

    Example:
    --------
    ```
    final_file1 = XYZ_value('path_to\LOCPOT', 'LOCPOT', c1=-1, align_pot=True, plot=True)
    print(f"XYZ data saved for LOCPOT in {final_file1}")
    final_file2 = XYZ_value('path_to\CHGCAR', 'CHGCAR')
    print(f"XYZ data saved for CHGCAR in {final_file1}")
    ```
    """

    output_file = f"XYZ_{type_file}.dat"
    
    print ("File to convert to xyz format: ",FILE1)
    NATOMS, GRID, POINTS, alattvec, blattvec, clattvec, skiplines, maxrows = read_file_info(FILE1)
    NGX,NGY,NGZ = GRID
    
    LATTCURB, VOL, omega = compute_lattice_parameters(alattvec, blattvec, clattvec)

    LATTCURADO = (1.0) * np.dstack([alattvec, blattvec, clattvec])
    LATTCURADO = LATTCURADO[0]
    
    LZ = clattvec[2]

    df = read_data(FILE1, skiplines, maxrows, POINTS)
    CHG1 = df.to_numpy().flatten('C')[0:POINTS]     

    #CHG1 = process_file(FILE1, skiplines, maxrows, POINTS)
    GHGtem = reshape_xyz(NGX,NGY,NGZ, CHG1)

    if type_file == 'CHGCAR':
        CHG = GHGtem/omega
    if type_file == 'LOCPOT': 
        CHG = GHGtem*(c1)
    
        AVEGPOTZcorr = compute_planar_average_nz(CHG, NGX, NGY, NGZ, 'z')
        first_value_AVEGPOTZcorr = AVEGPOTZcorr[0]
     
        if align_pot:
            CHG = CHG - first_value_AVEGPOTZcorr

        AVEGPOTZ = compute_planar_average_nz(CHG, NGX, NGY, NGZ, 'z')

        if plot:
            plt.plot(AVEGPOTZ)
            plt.xlabel('Z-direction')
            plt.ylabel('Planar Average of LOCPOT')
            plt.title('Electrostatic Potential')
            plt.savefig('PA_MEP.svg', format='svg', bbox_inches='tight')
            plt.show()

    
    final_file, rho = write_xyz(CHG, NGX, NGY, NGZ, LATTCURADO, omega, output_file)
    return final_file

def Perturbative_point(FILE1,FILE2,q,N):
    """
    Computes a a perturbative expansion of the energy, which is given by ΔU(r) = qΦ(r) - qΔNvf⁺⁻(r)

    Parameters:
    -----------
    FILE1 : str
        Path to the LOCPOT file with Electrostatic potential
    FILE2 : str
        Path to the LOCPOT file with Fukui potential vf⁺⁻(r).
    q : float
        Charge of active site.
    N : float
        Change in the number of electrons ΔN

    Returns:
    --------
    final_file : str
        Path to the output file containing the computed perturbative model potential in VASP format.

    Notes:
    ------
    - The charge density grids in the two input files must have the same dimensions.
    - The output potential is written in VASP format as 'MODELPOT.LOCPOT'.
    - The function assumes the grids are orthogonal, and the file format is consistent with standard VASP charge density files.

    Example:
    --------
    ```
    final_file = Perturbative_point('path_to/LOCPOT', 'path_to/LOCPOT_fukui', q=1.0, N=0.5)
    print(f"Perturbative model potential saved in {final_file}")
    ```
    """

    print ("FILE1: ",FILE1)
    NATOMS, GRID, POINTS, alattvec, blattvec, clattvec, skiplines, maxrows = read_file_info(FILE1)
    NGX,NGY,NGZ = GRID

    print ("FILE2: ",FILE2)
    NATOMS2, GRID2, POINTS2, alattvec2, blattvec2, clattvec2, skiplines2, MAXROWS2 = read_file_info(FILE2)
    
    if POINTS != POINTS2:
        print("The files have different grids")
        return
    
    
    df = read_data(FILE1, skiplines, maxrows, POINTS)
    CHG1 = df.to_numpy().flatten('C')[0:POINTS]  
    #CHG1 = process_file(FILE1,skiplines,maxrows, POINTS)
    CHG1 = CHG1.astype(np.float64)
    print('CHG1',CHG1)
    
    df = read_data(FILE2, skiplines, maxrows, POINTS)
    CHG2 = df.to_numpy().flatten('C')[0:POINTS]  
    #CHG2 = process_file(FILE2,skiplines2, MAXROWS2, POINTS)
    CHG2 = CHG2.astype(np.float64)
    print('CHG2',CHG2)

    
    print("")
    print("Just a few seconds.")

    CHGtem1 = reshape_xyz(NGX, NGY, NGZ, CHG1)
    CHGtem2 = reshape_xyz(NGX, NGY, NGZ, CHG2)


    CHGSUM = q*CHGtem1*(-1) - N*q*CHGtem2*(-1)
    

    AVEGPOTZcorr = compute_planar_average_nz(CHGSUM, NGX, NGY, NGZ, 'z')
    ##correción y alineación al vacío
    first_value_AVEGPOTZcorr = AVEGPOTZcorr[0]
    
    
    CHG = CHGSUM - first_value_AVEGPOTZcorr

    LOCPOTtem =  np.zeros(NGX*NGY*NGZ)
    for N1 in range(NGZ):
        for N2 in range(NGY):
            for N3 in range(NGX):
                LOCPOTtem[N1*NGX*NGY + N2*NGX + N3] = CHG[N3,N2,N1]


    final_file = write_fukui_file(FILE2, NATOMS, LOCPOTtem, 'MODELPOT.LOCPOT') 
    return final_file

def min_max(FILE1, extrema_point ):
    """
    Identifies and extracts the local minima or maxima from a 3D potential field.

    Parameters:
    -----------
    FILE1 : str
        Path to the input file containing the potential data (e.g., CHGCAR or LOCPOT) in VASP format.
    extrema_point : str
        Specifies the type of extremum to find: 'min' for minima or 'max' for maxima.

    Returns:
    --------
    name_file : str
        Path to the output file where the coordinates and values of the identified extrema are saved.
    m_points : list of tuples
        List of the coordinates and corresponding values of the local minima or maxima.

    Description:
    ------------
    This function processes the potential data from the input file (`FILE1`) and identifies the local minima or maxima
    in the 3D potential field. The extrema are detected using a second-order finite difference method (order=2). 
    The function then calculates the corresponding coordinates in real space using the lattice vectors and saves the 
    results to a text file with the name `extrema_point.txt` (where `extrema_point` is either 'min' or 'max').

    Notes:
    ------
    - The input file (`FILE1`) must contain a 3D potential field in a format compatible with VASP charge density files.
    - The lattice vectors are used to convert the 3D grid indices to real-space coordinates.
    - The output file contains the coordinates and values of the identified extrema.
    
    Example:
    --------
    ```
    name_file, extrema_points = min_max('path_to/LOCPOT', extrema_point='min')
    print(f"Local minima saved in {name_file}")
    ```
    """

    NATOMS, GRID, POINTS, alattvec, blattvec, clattvec, skiplines, maxrows = read_file_info(FILE1)
    POT = process_file(FILE1, skiplines, maxrows, POINTS)
    POT = np.array(POT).astype(float)
    POT = POT.reshape(GRID[2], GRID[1], GRID[0])
    coords, values = detect_local_extrema_3D(POT, order=2, extrema_type=extrema_point)
    m_points = calcular_xyz_val(coords, values, GRID, alattvec, blattvec, clattvec)
    name_file = f"{extrema_point}.txt"
    write_formatted_data_localm(name_file, m_points)
    return name_file,m_points

def visual_modelpot(file1, file2):
    """
    Plot heatmap with respect to the X and Y distance, and a color bar is added to indicate the energy 
      difference (`ΔU_int`).
    Parameters:
    -----------
    file1 : str
        Path to the input file containing the charge density data of neutral slab (e.g., CHGCAR) in VASP format.
    file2 : str
        Path to the input file containing the energy obtained by perturbative expansion (e.g., MODELPOT) in VASP format.

    Returns:
    --------
    filtrados : list of tuples
        A list of filtered data points, where each tuple contains (x, y, z, value) for the data that satisfies the filter conditions.
        The function generates a heatmap and saves it as "heatmap_MODELPOT.png" and the data as "filtered_data.txt". 
        The plot is displayed using matplotlib.
    Description:
    ------------
    This function filters data to exclude values that are not close to a density of 1e-3 and do not lie within the upper half of the slab 
    (refer to the filter_values() function for more details). It then creates a heatmap based on a 2D histogram of the filtered data, 
    where the X and Y axes represent spatial coordinates and the color intensity corresponds to energy differences. The heatmap is displayed 
    and saved as an image file ("heatmap_MODELPOT.png"), and the filtered data points are written to "filtered_data.txt".

    Notes:
    ------
    - The function assumes that both input files (`file1` and `file2`) are in the correct VASP format for charge density 
      (CHGCAR) and electrostatic potential (LOCPOT) files.

    Example:
    --------
    ```
    visual_modelpot('CHGCAR_file', 'MODELPOT')
    ```
    This will create a heatmap of the electrostatic potential difference between the two files.
    """

    # Files to compare
    archivo1 = XYZ_value(file1, 'CHGCAR')
    archivo2 = XYZ_value(file2, 'LOCPOT')

    print('Reading: ', archivo1,'and', archivo2, 'to plot')
    NATOMS, GRID, POINTS, alattvec, blattvec, clattvec, skiplines, maxrows = read_file_info(file1)
    NGX,NGY,NGZ = GRID
    z_sup = np.linalg.norm(clattvec)*0.5

    datos1 = read_xyzval(archivo1)
    datos2 = read_xyzval(archivo2)

    compare_columns(datos1, datos2)

    filtrados = filter_values(datos1, datos2, z_sup)

    np.savetxt('heatmap_data.txt', filtrados, fmt='%f', header="X Y Z Delta_U_int", comments='')

    # Values for heatmap
    x_vals = [item[0] for item in filtrados]
    y_vals = [item[1] for item in filtrados]
    valores = [item[3] for item in filtrados]

    heatmap_data, x_edges, y_edges = np.histogram2d(x_vals, y_vals, bins=(NGX, NGY), weights=valores)

    x_range = x_edges[-1] - x_edges[0]
    y_range = y_edges[-1] - y_edges[0]

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data.T, cmap='jet_r', origin='lower',
               extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
               aspect=x_range/y_range)  # Escalar ejes según el rango


    cbar = plt.colorbar()
    cbar.set_label(r"$\Delta U_{int}$", fontsize=14, fontweight='bold') 
    cbar.ax.tick_params(labelsize=12) 


    plt.title(r"$\Delta U_{int}$ at  $\rho =10^{-4}$ $a_{0}^{-3}$", size=16, family='sans-serif')
    plt.xlabel(r"X (Å)",size=14, family='sans-serif')
    plt.ylabel(r"Y (Å)", size=14,family='sans-serif')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig("heatmap_MODELPOT.png")
    plt.show()
    return filtrados
  
    
############################################################
#                        Main Menu                         #
############################################################

def main_menu():
        while True:

            #******Print Header*****
            header = """
                                            8888888                                   8888888
8888888888                  8888888888      888          888b    888                      888 
888                         888             888          8888b   888                      888 
888                         888             888          88888b  888                      888 
8888888         888888      8888888         888          888Y88b 888    888  888          888 
888                         888             888          888 Y88b888    888  888          888 
888             888888      888             888          888  Y88888    Y88  88P          888 
888                         888             888          888   Y8888 d8b Y8bd8P           888 
8888888888                  8888888888      888          888    Y888 88P  Y88P            888 
                                            8888888                  8P               8888888        
                                                                     "                        
                                                                                              
"""

            print(header)
            print("FukuiGrid -- A useful tool for Conceptual DFT in Solid-State.")
            print("Version 1.0, release date: January-2025")
            print("Developers: Javiera Cabezas-Escares, Nicolas F.Barrera, Prof. Carlos Cardenas -- ")
            print("            TheoChemPhys Group, University of Chile.")
            print("            https://github.com/cacarden/FukuiGrid")
            print("")
            print("")
            print("")
            print("               ***** Main Menu *****               ")
            print("1 -- Fukui Function via Interpolation")
            print("2 -- Fukui Potential via Electrodes")
            print("3 -- Fukui Potential via SCPC")
            print("4 -- Process Grid Data")
            print("5 -- Perturbative Expansion")
            print("6 -- Exit")
            print("")
            option = input("Choose an option: ")

            if option == "1":
                print("\nChose option 1: Fukui Function via Interpolation.\n")
                print("11 Electrophilic Fukui function f^-(r).")
                print("12 Nucleophilic  Fukui function f^+(r).")

                option1 = input("Choose an option: ")
                if option1 == "11":
                    print("Name CHGCAR files with \u03B4N: -0.15, -0.10, -0.05, and 0.0")
                    FILE1 = input("Enter name of file 1: ")
                    FILE2 = input("Enter name of file 2: ")
                    FILE3 = input("Enter name of file 3: ")
                    FILE4 = input("Enter name of file 4: ")
                    print("\nThis will take a few seconds.\n")

                    Fukui_interpolation(FILE1, FILE2, FILE3, FILE4, [-0.15, -0.10, -0.05, 0.0])

                    continue
    
                if option1 == "12":
                    print("Name CHGCAR files with \u03B4N: 0.0, +0.05, +0.010, and +0.15")
                    FILE1 = input("Enter name of file 1: ")
                    FILE2 = input("Enter name of file 2: ")
                    FILE3 = input("Enter name of file 3: ")
                    FILE4 = input("Enter name of file 4: ")

                    print("\nThis will take a few seconds.\n")

                    Fukui_interpolation(FILE1, FILE2, FILE3, FILE4, [0.0, 0.05, 0.10, 0.15])
                    
                    continue

                else:
                    print("Invalid option. Please select an option from the menu.")

            elif option == "2":
                print("You selected option 2: Fukui Potential via Electrodes' method")
                print("\nName CHGCAR file of charge density of the neutral slab.")
                FILE0 = input("Enter file name: ")
                print("\nName CHGCAR file of Fukui function.")
                FILE1 = input("Enter file name: ")
                print("\nDielectric constant value.")
                Epsilon = float(input("Value: ")) 
                fukui_electrodes(FILE0, FILE1, Epsilon)
                
                continue
            
            elif option == "3":
                print("You selected option 3: Fukui Potential via SCPC")
                print("")
                print("")
                print("31 -- Electrophilic Fukui function v_f^-(r).")
                print("32 -- Nucleophilic Fukui function v_f^+(r).")
                print("")

                option3 = input("Choose an option: ")
                if option3 == "31":
                    print("Name CHGCAR file of charge density of the neutral slab.")
                    FILE0 = input("Enter file name: ")
                    print("\nName CHGCAR file of Fukui function.")
                    FILE1 = input("Enter file name: ")
                    print("\nName SCPC correcton file.")
                    FILE2 = input("Enter file name: ")

                    fukui_SCPC(FILE0,FILE1,FILE2,-1)

                    continue
                
                if option3 == "32":
                    print("Name CHGCAR file of charge density of the neutral slab.")
                    FILE0 = input("Enter file name: ")
                    print("\nName CHGCAR file of Fukui function.")
                    FILE1 = input("Enter file name: ")
                    print("\nName SCPC correcton file.")
                    FILE2 = input("Enter file name: ")

                    fukui_SCPC(FILE0,FILE1,FILE2,1)
                    
                    continue

                else:
                    print("Invalid option. Please select an option from the menu.")

            elif option == "4":
                print("You selected option 4: Process Grid Data")
                print("")
                print("")
                print("41 -- Add:            Add two CHGCAR or LOCPOT.")
                print("42 -- Subtract:       Subtracts two CHGCAR or LOCPOT.")
                print("43 -- Scale:          Multiply a CHGCAR or POTCAR by a scalar.")
                print("44 -- Add a constant: Adds a constant to a CHGCAR or LOCPOT file. ")
                print("45 -- Planar Average: Compute the planar average of CHGCAR or LOCPOT in the X, Y, or Z direction.")
                print("46 -- Value-xyz:      Convert a CHGCAR or LOCPOT into a simple list: X, Y, Z, Value format.")
                print("47 -- Max:            Find Local Maxima for a LOCPOT file")
                print("48 -- Min:            Find Local Minima for a LOCPOT file")
                print("")
                print("")

                option4 = input("Choose an option: ")
                if option4 == "41":
                    print("Name CHGCAR or LOCPOT files to add: ")
                    FILE1 = input("Enter name of file 1: ")
                    FILE2 = input("Enter name of file 2: ")
                    print("Enter the constants c1 and c2 according to FILE1*c1 + FILE2*c2: ")
                    c1 = float(input("c1: "))
                    c2 = float(input("c2: "))
                    print("\nThis will take a few seconds.\n")
                    
                    lineal_operation(FILE1,FILE2,c1,c2,0)
                   
                    # Change the file name
                    old_filename = "CHGCARSUM"
                    new_filename = "CHGCAR_SUM"
                    os.rename(old_filename, new_filename)

                    continue

                if option4 == "42":
                    print("Name CHGCAR or LOCPOT files to subtract: ")
                    FILE1 = input("Enter name of file 1: ")
                    FILE2 = input("Enter name of file 2: ")
                    print("\nThis will take a few seconds.\n")

                    lineal_operation(FILE1,FILE2,1,-1,0)
                    
                    # Change the file name
                    old_filename = "CHGCARSUM"
                    new_filename = "CHGCAR_DIFF"
                    os.rename(old_filename, new_filename)

                    continue

                if option4 == "43":
                    print("Name CHGCAR or LOCPOT files to scale: ")
                    FILE1 = input("Enter name of file 1: ")
                    FILE2 = FILE1
                    print("")
                    print("Enter the scale factor c:")
                    c1 = float(input("c: "))

                    lineal_operation(FILE1,FILE2,c1,0,0)

                    # Change the file name
                    old_filename = "CHGCARSUM"
                    new_filename = "CHGCAR_SCALE"
                    os.rename(old_filename, new_filename)
                    
                    continue
                
                if option4 == "44":
                    print("Name CHGCAR or LOCPOT files to add it a constant to:")
                    FILE1 = input("Enter name of file 1: ")
                    FILE2 =  FILE1
                    print("")
                    print("Enter the constant c:")
                    c3 = float(input("c: "))

                    lineal_operation(FILE1,FILE2,1,0,c3)

                    # Change the file name
                    old_filename = "CHGCARSUM"
                    new_filename = "CHGCAR_constant"
                    os.rename(old_filename, new_filename)

                    continue

                if option4 == "45":
                   print("")
                   print("451 -- Planar Average for a CHGCAR file.") 
                   print("452 -- Planar Average for a LOCPOT file.")
                   print("")
                   option45 = input("Choose an option: ")
                   option45_axis = input("Select x, y, or z: ")
                   
                   if option45 == "451":
                       print("")
                       FILE1 = input("Enter name of CHGCAR file: ")
                       planar_average(FILE1,'CHGCAR',option45_axis)

                       continue

                   if option45 == "452":
                       print("")
                       FILE1 = input("Enter name of LOCPOT file: ")
                       planar_average(FILE1, 'LOCPOT',option45_axis)

                       continue

                if option4 == "46":
                    print("\n461 -- XYZ-value for a CHGCAR file.\n")
                    print("\n462 -- XYZ-value for a LOCPOT file.\n")
                    option46 = input("Choose an option:")

                    if option46 == "461":
                        print("")
                        FILE1 = input("Enter name of CHGCAR file: ")
                        XYZ_value(FILE1, 'CHGCAR')

                        continue
                    
                    if option46 == "462":
                        print("")
                        FILE1 = input("Enter name of LOCPOT file: ")
                        print("")
                        print("Do you want to align potential to zero in the vacuum?")
                        option462 = input("yes or no: ")

                        if option462 == "yes" or option462 == "YES" or option462 == "y":
                            print("")
                            print("Do you want to write the data in the chemical convention (i.e. take the negative)?")
                            option462y = input("yes or no: ")
                            if option462y == "yes" or option462y == "YES" or option462y == "y":
                                print("")
                                print("Do you want to generate a plan average plot?")
                                option462p = input("yes or no: ")
                                if option462p == "yes" or option462p == "YES" or option462p == "y":
                                    XYZ_value(FILE1, 'LOCPOT', -1,align_pot=True,plot=True)
                                    continue
                                elif option462p == "no" or option462p == "NO" or option462p == "n":
                                    XYZ_value(FILE1, 'LOCPOT', -1,align_pot=True)
                                    continue

                            elif option462y == "no" or option462y == "NO" or option462y == "n":
                                print("")
                                print("Do you want to generate a plan average plot?")
                                option462p = input("yes or no: ")
                                if option462p == "yes" or option462p == "YES" or option462p == "y":
                                    XYZ_value(FILE1, 'LOCPOT', 1,align_pot=True,plot=True)
                                    continue
                                elif option462p == "no" or option462p == "NO" or option462p == "n":
                                    XYZ_value(FILE1, 'LOCPOT', 1,align_pot=True)
                                    continue

                        if option462 == "no" or option462 == "NO" or option462 == "n":
                            print("")
                            print("Do you want to write the data in the chemical convention (i.e. take the negative)?")
                            option462n = input("yes or no: ")
                            if option462n == "yes" or option462 == "YES" or option462 == "y":
                                print("")
                                print("Do you want to generate a plan average plot?")
                                option462p = input("yes or no: ")
                                if option462p == "yes" or option462p == "YES" or option462p == "y":
                                    XYZ_value(FILE1, 'LOCPOT', -1,align_pot=False,plot=True)
                                    continue
                                elif option462p == "no" or option462p == "NO" or option462p == "n":
                                    XYZ_value(FILE1, 'LOCPOT', -1,align_pot=False)
                                    continue
                            
                            elif option462n == "no" or option462n == "NO" or option462n == "n":
                                 print("")
                                 print("Do you want to generate a plan average plot?")
                                 option462p = input("yes or no: ")
                                 if option462p == "yes" or option462p == "YES" or option462p == "y":
                                    XYZ_value(FILE1, 'LOCPOT', 1,align_pot=False,plot=True)
                                    continue
                                 elif option462p == "no" or option462p == "NO" or option462p == "n":
                                    XYZ_value(FILE1, 'LOCPOT', 1,align_pot=False)
                                    continue
                
                if option4 == "47":
                    print("Name CHGCAR or LOCPOT files to found maxima: ")
                    FILE = input("Enter name of file: ")
                    min_max(FILE, 'max')
                
                if option4 == "48":
                    print("Name CHGCAR or LOCPOT files to foun minima: ")
                    FILE = input("Enter name of file: ")
                    min_max(FILE, 'min')

            elif option == "5":

                print("\nYou selected Perturbative Expansion.")
                mu_pm = "\u03BC\u207A\u207B"
                #Delta_U = "\u0394U = " + mu_pm + "\u0394N + q\u03A6(r) - q\u0394Nvf\u207A\u207B(r)"
                #print(Delta_U)
                Delta_U = "\u0394U(r) = q\u03A6(r) - q\u0394Nvf\u207A\u207B(r)"
                print(Delta_U)
                print("\nName of LOCPOT file with Electrostatic potential \u03A6(r).")
                FILE0 = input("Enter name of LOCPOT: ")
                print("\nName of LOCPOT file with Fukui potential vf\u207A\u207B(r).")
                FILE1 = input("Enter name of LOCPOT: ")
                print("\nEnter the change in the number of electrons \u0394N: ")
                N = float(input("\u0394N: "))
                print("\nEnter the charge q of active site: ")
                q = float(input("q: "))
                print("")
                print("Do you want a heat map of \u0394U(r)? ")
                option51 = input("yes or no: ")
                if option51 == "yes" or option51 == "YES" or option51 == "y":
                    print("Name CHGCAR file of charge density of the neutral slab.")
                    FILE_CHGCAR = input("Enter file name: ")
                    modelpot = Perturbative_point(FILE0,FILE1,q,N)
                    visual_modelpot(FILE_CHGCAR,modelpot)
                    continue
                elif option51 == "no" or option51 == "NO" or option51 == "n":
                    Perturbative_point(FILE0,FILE1,q,N)

                continue

            elif option == "6":
                print("You selected option 6: Goodbye!")
                sys.exit()

            else:
                print("Invalid option. Please select an option from the menu.")

if __name__ == "__main__":
        main_menu()
