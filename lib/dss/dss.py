import torch
import numpy as np
import subprocess 
import os
from .gslib import Gslib


class DSSClient():
    def __init__(self, args, real_fac_model=None, ipmin=None, ipmax=None, ipzones=None):
        if not hasattr(args, "in_folder"):
            args.in_folder = ""
        if not hasattr(args, "type_of_FM"):
            args.type_of_FM = ""
            
        self.inf= f'{args.project_path}/{args.in_folder}'
        self.ouf= f'{args.outdir}'
        self.nx= args.nx
        self.nz= args.nz
        self.var_N_str= args.var_N_str if hasattr(args, 'var_N_str') else 0
        self.var_nugget= args.var_nugget if hasattr(args, 'var_nugget') else 0
        self.var_type= args.var_type  if hasattr(args, 'var_type') else 0
        self.var_ang= args.var_ang  if hasattr(args, 'var_ang') else 0
        self.var_range= args.var_range   if hasattr(args, 'var_range') else 0
        self.null_val= args.null_val if hasattr(args, 'null_val') else -9999.99
        
        if args.type_of_FM == "fullstack":
            self.simulations= torch.zeros((args.nz, args.nx))
        else:
            pass

        try:
            self.ipmin=ipmin
            self.ipmax=ipmax
            self.ipzones=ipzones
        except: 
            raise TypeError('No Ip values bounds per facies provided')

        if real_fac_model!=None: 
            if args.ip_type==0:
                self.real_model= self.det_Ip(real_fac_model)
            elif args.ip_type==1:
                real_fac_model = (real_fac_model+1)*0.5
                self.real_model= self.run_dss(real_fac_model)[0,None,:]
        
            
    def det_Ip(self, facies_model):
        if (facies_model<0).any(): facies_model= (facies_model+1)*0.5
        self.simulations = (self.ipmin-self.ipmax)*facies_model+self.ipmax
        return self.simulations
    
    
    def writeallfac_dss(self, facies_mod):
        facies_mod= np.round(facies_mod.reshape(-1,facies_mod.shape[-2]*facies_mod.shape[-1]))
        for f in range(facies_mod.shape[0]):
            with open(self.ouf+f'/dss/Facies_model_{f+1}.out','w') as fid:
                fid.write('Facies\n')
                fid.write('1\n')
                fid.write('Facies\n')
                fid.write('\n'.join(facies_mod[f].astype(int).astype(str).tolist()))
        return None
        
    def write_parfile(self,s_f,nsim):
        text=[]
        text.append(f'[ZONES]\nZONESFILE = {self.ouf}/dss/Facies_model_{s_f+1}.out  # File with zones\nNZONES={len(self.ipzones)}  # Number of zones\n\n')
        for fac in range(len(self.ipzones)):
            text.append(f'[HARDDATA{fac+1}]\nDATAFILE = {self.inf}/Ip_zone{fac}.out  # Hard Data file\n')
            text.append('COLUMNS = 4\nXCOLUMN = 1\nYCOLUMN = 2\nZCOLUMN = 3\nVARCOLUMN = 4\nWTCOLUMN = 0\n')
            text.append(f'MINVAL = {self.ipzones[fac][0]}  # Minimun threshold value\nMAXVAL = {self.ipzones[fac][1]}  # Minimun threshold value\n')
            text.append('USETRANS = 1\nTRANSFILE = Cluster.trn  #Transformation file\n\n')
        text.append(f'[HARDDATA]\nZMIN = {self.ipmin}  # Minimum allowable data value\nZMAX = {self.ipmax}  # Maximum allowable data value\nLTAIL = 1\nLTPAR = {self.ipmin}\nUTAIL = 1\nUTPAR = {self.ipmax}\n\n')
        text.append(f'[SIMULATION]\nOUTFILE = {self.ouf}/dss/ip_real  # Filename of the resultant simulations\nNSIMS = {nsim}  # Number of Simulations to generate \nNTRY = 10\nAVGCORR = 1\nVARCORR = 1\n\n')
        text.append(f'[GRID]\nNX = {self.nx}\nNY = 1\nNZ = {self.nz}\nORIGX = 1\nORIGY = 1\nORIGZ = 1\nSIZEX = 1\nSIZEY = 1\nSIZEZ = 1\n\n')
        text.append(f'[GENERAL]\nNULLVAL = {self.null_val} \nSEED = {self.seed}\nUSEHEADERS = 1\nFILETYPE = GEOEAS\n\n')
        text.append(f'[SEARCH]\nNDMIN = 1\nNDMAX = 32\nNODMAX = 12\nSSTRAT = 1\nMULTS = 0\nNMULTS = 1\nNOCT = 0\nRADIUS1 = {self.nx}\nRADIUS2 = 1\nRADIUS3 = {self.nz}\nSANG1 = 0\nSANG2 = 0\nSANG3 = 0\n\n')
        text.append(f'[KRIGING]\nKTYPE = {self.krig_type}  # Kriging type: 0=simple,1=ordinary,2=simple with locally varying mean, 3=external drif, 4=collo-cokrig global CC,5=local CC (KTYPE)\n')
        text.append(f'COLOCORR = 0.75\nSOFTFILE = {self.sec_var_file}\nLVMFILE = No File\nNVARIL = 1\nICOLLVM = 1\nCCFILE = {self.local_corr_file}\nRESCALE = 1\n\n')
        
        for fac in range(len(self.ipzones)):
            text.append(f'[VARIOGRAMZ{fac+1}]\nNSTRUCT = {self.var_N_str[fac]}  # Number of semivariograms structures\nNUGGET = {self.var_nugget[fac]}  # Nugget constant\n\n')
            for struct in range(self.var_N_str[fac]):
                text.append(f'[VARIOGRAMZ{fac+1}S{struct+1}]\nTYPE = {self.var_type[fac][struct]}\nCOV = 1\n')
                text.append(f'ANG1 = {self.var_ang[fac][0]}\nANG2 = 0\nANG3 = {self.var_ang[fac][1]}\n')
                text.append(f'AA = {self.var_range[fac][struct][0]}\nAA1 = 1\nAA2 = {self.var_range[fac][struct][1]}\n\n')
            text.append(f'[BIHIST{fac+1}]\nUSEBIHIST = 0\nBIHISTFILE = No File\nNCLASSES = 30\nAUXILIARYFILE = No File\n\n')
        text.append('[DEBUG]\nDBGLEVEL = 1\nDBGFILE = debug.dbg\n\n')
        text.append(f'[COVTAB]\nMAXCTX = {self.nx}\nMAXCTY = 1\nMAXCTZ = {self.nz}\n\n')
        text.append('[BLOCKS]\nUSEBLOCKS = 0\nBLOCKSFILE = NoFile\nMAXBLOCKS= 100\n\n[PSEUDOHARD]\nUSEPSEUDO = 0\nBLOCKSFILE = No File\nPSEUDOCORR = 0\n')
                
        text= ''.join(text)
        
        with open(self.inf+'/ssdir.par', 'w') as ssdir:
            ssdir.write(text)
            

    def run_dss(self, facies_mod, i, args):
        self.simulations= torch.zeros((facies_mod.shape[0], 1, facies_mod.shape[2], facies_mod.shape[3]))
        if facies_mod.min() < 0: facies_mod= (facies_mod.detach().cpu().numpy()+1)*0.5
        else: facies_mod= facies_mod.detach().cpu().numpy()
        self.writeallfac_dss(facies_mod)
        
        if i < 1:
            self.sec_var_file='No file'
            self.local_corr_file='No file'
            self.krig_type= 0
        else:
            self.krig_type= 5
            self.sec_var_file=f'{self.ouf}/aux_ip.out'
            self.local_corr_file=f'{self.ouf}/aux_simil.out'    

        
        for s_f in range(facies_mod.shape[0]):
            self.seed= np.random.randint(1000,100000,1)[0]
            self.write_parfile(s_f,args.n_sim_ip)
            
            if os.name == 'nt':
                dss_bin = f'{self.inf}/dss_nt.exe'
            else:
                dss_bin = f'{self.inf}/dss_unix'
            
            subprocess.run(args=[dss_bin, f'{self.inf}/ssdir.par'], stdout=subprocess.DEVNULL)
            
            ssimss= np.zeros((args.n_sim_ip,1,self.nz,self.nx))
            for ssi in range (0,args.n_sim_ip):
                ssimss[ssi]= np.reshape(Gslib().Gslib_read(f'{self.ouf}/dss/ip_real_{ssi+1}.out').data.squeeze(),
                              (self.simulations.shape[-3], self.simulations.shape[-2], self.simulations.shape[-1]))

            self.simulations[s_f]= torch.mean(torch.from_numpy(ssimss), axis=0)
                    
            
        return self.simulations
