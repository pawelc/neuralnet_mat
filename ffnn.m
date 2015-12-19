classdef ffnn < handle
    %NN Neural network
    %   Contains NN structure
    
    properties
        %neurons in each layer
        architecture
        
        %% number of layers
        L
        
        %weihts, initialize to random values
        W
        
        %cell array containing linear part of output vector for each layer
        I
        
        %cell array containing output vector for each layer
        Z
        
    end
    
    methods
        function net = ffnn(architecture)
            net.architecture = architecture;
            net.L=size(architecture,1);
            
            net.W=cell(1,net.L);
            for l = 2:net.L
                %adding 1 for bias
                net.W{l}=(rand(architecture(l),architecture(l-1)+1)*2-1)/sqrt(architecture(l-1)+1);
            end
            
            net.I=cell(1,net.L);
            for l=1:net.L
                net.I{l}=zeros(architecture(l),1);
            end
            
            net.Z=cell(1,net.L);
            for l=1:net.L
                net.Z{l}=[1;zeros(architecture(l),1)];
            end
        end
        
        function forward(obj,x)
            obj.I{1} = x;
            obj.Z{1}(2:end) = x;
            for l=2:obj.L
                obj.I{l}=obj.W{l}*obj.Z{l-1};
                obj.Z{l}(2:end)=tanh(obj.I{l});
            end
        end
        
        function out = output(obj)
            out=obj.Z{obj.L}(2:end);
        end
        
        function rmse = rmse(obj,X,Y)
            rmse = 0;
            %RMSE Computes RMSE
            for k=1:size(X,1)
                obj.forward(X(k,:)');
                out = obj.output();
                diff = out-Y(k,:)';
                rmse = rmse + diff'*diff;
            end
            rmse=sqrt(rmse/size(X,1));
        end
    end
    
end

