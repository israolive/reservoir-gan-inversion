import numpy as np


class Gslib:
    def __init__(self):
        self.filename ='Null'
        self.title ='Null'
        self.n_properties = 0
        self.prop_names = []
        self.n_lines_head = 0
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.data = []


    def __repr__(self):
        if self.n_properties>1:
            return f"<gslib file_name={self.filename} n_properties={self.n_properties} properties={self.prop_names} />"
        elif self.n_properties==1:
            return f"<gslib file_name={self.filename} properties={self.prop_names} />"
        else:
            return "<gslib />"


    def Gslib_read(self, filename, order_z='no', overwrite_file= 'no', zmax=0):
        #Reads a Gslib file and creates a Gslib Instance
        
        self.filename= filename
        
        with open(self.filename, 'r+') as f:
            self.title= f.readline()
        
            self.n_properties= int(f.readline().split()[0])
        
            self.prop_names=[]
            for i in range(self.n_properties):
                self.prop_names.append(f.readline().split()[0])
        
            self.n_lines_head = self.n_properties + 2
            
            # Read data using numpy
            # Gslib format uses space separation
            try:
                self.data = np.loadtxt(self.filename, skiprows=self.n_lines_head)
            except Exception as e:
                # If reading fails (e.g. empty file or malformed), init empty
                print(f"Error reading gslib data: {e}")
                self.data = np.array([])
            
            if order_z=='yes':
                z_col_idx = 2
                if self.data.ndim == 2 and self.data.shape[1] > z_col_idx:
                    self.data[:, z_col_idx] = self.data[:, z_col_idx] * -1 + zmax + 1
                
            if overwrite_file == 'yes':
                f.seek(0)
                f.write(self.title)
                f.write(f'{self.n_properties}\n')
            
                for i in self.prop_names:
                    f.write(f'{i}\n')
                
                f.truncate()
                np.savetxt(f, self.data, fmt='%.6f', delimiter='\t')

        return self

    def Gslib_write(self, filename, prop_names, data, nx, ny, nz, folder, title='null'):
        # writes a Gslib file and returns a Gslib Instance
        self.filename=filename
        
        if title== 'null':
            self.title=filename
        else: 
            self.title=title
        
        if type(prop_names) is list or type(prop_names) is np.ndarray:
            self.prop_names=prop_names
        else:
            self.prop_names=[prop_names]
            
        self.n_properties=len(self.prop_names)
        self.nx=nx
        self.ny=ny
        self.nz=nz
        self.data=data
        self.path= f'{folder}/{filename}.out'
        
        # Writes the header in the file
        with open(self.path,'w') as f:
            f.write(f'{self.title}\n')
            f.write(f'{self.n_properties}\n')
            for i in range(self.n_properties):
                f.write(f'{self.prop_names[i]}\n')
        
        if type(data) is tuple or type(data) is list:
            data=np.asarray(data)

        # Writes data to file
        with open(self.path,'a') as f:

            if type(data) is np.ndarray:        
                
                if self.n_properties==1:
                    if data.shape == (1, self.nx*self.ny*self.nz): data= data.T #if data is a 1D array,it is transposed for writing in a single column
                    elif data.shape == (self.nx*self.ny*self.nz,): pass #If data is already in one column
                    elif data.shape == (self.nx,self.ny,self.nz): data= np.reshape(data, newshape=(nx*ny*nz,1), order='F') #if data is not a 1D array, but a 3D grid, changes shape to the array for writing
                    np.savetxt(f, data, fmt='%.6f')
                
                elif self.n_properties>1:
                    
                    if data[0].shape == (1, self.nx*self.ny*self.nz): data= data.T #if is an array of dimension= n_properties
                    elif data[0].shape == (self.ny,) or data[0].shape== (self.n_properties,) or data[0].shape == (self.nx*self.ny*self.nz, 1): pass #if data is an array of dimension= n_properties - for well logs
                    elif data[0].shape == (self.nx,self.ny,self.nz): 
                        for i in self.n_properties: data[i]= np.reshape(data[i], newshape=(nx*ny*nz,1), order='F')

                    np.savetxt(f, data,fmt='%.6f')
                    
                else:
                    return "Cannot read input data correctly"
                
            else:
                return "Input data type not recognized"

        return self
    
    def Gslib_writethis(self, path):
        """
        Use this module only when you need to write a Gslib class that has been already defined
        """
        if self.title == "null":
            self.title=self.filename
        
        with open(path, "w") as f:
            f.write(f'{self.title}\n')
            f.write(f'{self.n_properties}\n')
            for i in range(self.n_properties):
                f.write(f'{self.prop_names[i]}\n')
        
        # Convert the data to numpy array (only for writing, original data type will be kept in the Gslib instance)
        if type(self.data) is tuple or type(self.data) is  list:
            self.data=np.asarray(self.data)

        # Writes data to file
        with open(path, "a") as f:
            if type(self.data) is np.ndarray:        
                
                if self.n_properties==1 and self.data.shape == (1, self.nx*self.ny*self.nz):
                    #if data is a 1D array,it is transposed for writing in a single column
                    self.data= self.data.T
                    np.savetxt(f, self.data, fmt='%.6f')
                    
                elif self.n_properties==1 and self.data.shape == (self.nx*self.ny*self.nz,):
                    #If data is already in one column
                    np.savetxt(f, self.data, fmt='%.6f')
                
                elif self.n_properties==1 and self.data.shape == (self.nx,self.ny,self.nz):
                    #if data is not a 1D array, but a 3D grid, changes shape to the array for writing
                    self.data= np.reshape(self.data, newshape=(self.nx*self.ny*self.nz,1), order='F')
                    np.savetxt(f, self.data,fmt='%.6f')
                
                elif self.n_properties>1 and self.data[0].shape == (1, self.nx*self.ny*self.nz): 
                    #if is an array of dimension= n_properties
                    self.data= self.data.T
                    np.savetxt(f, self.data,fmt='%.6f')
                
                elif self.n_properties>1 and self.data[0].shape == (self.nx*self.ny*self.nz, 1): 
                    #If data is already in n_properties columns
                    np.savetxt(f, self.data,fmt='%.6f')
                
                elif self.n_properties>1 and self.data[0].shape == (self.nx,self.ny,self.nz): 
                    #if input data is n_properties 3D grids, reshape each 3D grid in 1D array
                    for i in self.n_properties: 
                        self.data[i]= np.reshape(self.data[i], newshape=(self.nx*self.ny*self.nz,1), order='F')
                        
                    np.savetxt(f, self.data,fmt='%.6f')   
                
                else:
                    print ("Cannot read input data correctly")
                    return None
                
            else:
                print ("Data type not recognized")
                return None

        return None
