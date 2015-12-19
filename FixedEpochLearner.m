classdef FixedEpochLearner < handle
    %FIXEDEPOCHLEARNER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        %FFNN
        nn
        X
        
        Y
        
        U
        
        epochs     
        
        diagnostics
    end
    
    methods
        
        %% Constructor
        function l = FixedEpochLearner(nn)
            l.nn=nn;            
        end
        
        %% learn function
        function learn(obj)
            %init diagnostics
            obj.diagnostics.trainRmse = zeros(obj.epochs,1);
            
            obj.U = atanh(obj.Y);
            
            %% iterator over all data
            for epoch=1:obj.epochs
                for k=1:size(obj.X,1)                                        
                    %% forward phase
                    input=obj.X(k,:)';
                    output=obj.Y(k,:)';
                    %expected linear output
                    linOutput=obj.U(k,:)';
                    obj.nn.forward(input);                                        
                    obj.update(input,output,linOutput);
                end
                
                %comput RMSE
                trainRmse = obj.nn.rmse(obj.X,obj.Y);                
                
                obj.diagnostics.trainRmse(epoch)=trainRmse;
                
            end
        end
    end
    
    methods (Abstract)
        update(obj,input,output,linOutput)
    end
    
end

