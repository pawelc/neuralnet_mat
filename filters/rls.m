classdef rls < FixedEpochLearner
    %RLS learns using recursive least squares
    
    properties        
        %cell array containing error covariance matrices for weights of each neuron
        P        
        G
        D
    end
    
    methods
        function f = rls(nn)
            f@FixedEpochLearner(nn)                      
            
            f.P=cell(1,nn.L);
            f.G=cell(1,nn.L);
            f.D=cell(1,nn.L);
            for l=2:nn.L
                f.D{l}=zeros(nn.architecture(l),1);
                for n=1:nn.architecture(l)
                    f.P{l}{n}=eye(nn.architecture(l-1)+1)*0.01;                    
                    f.G{l}{n}=zeros(nn.architecture(l-1)+1,1);
                end
            end            
        end
        
        function update(obj,~,~,linOutput)                  
            %local gradient
            obj.D{obj.nn.L}=linOutput-obj.nn.I{obj.nn.L};
            for l=obj.nn.L-1:-1:2
                obj.D{l}=(1-obj.nn.Z{l}(2:end).^2).* (obj.nn.W{l+1}(:,2:end)'*obj.D{l+1});
            end
            
            for l=2:obj.nn.L
                for n=1:obj.nn.architecture(l)
                    obj.G{l}{n}=obj.P{l}{n}*obj.nn.Z{l-1}/(obj.nn.Z{l-1}'*obj.P{l}{n}*obj.nn.Z{l-1}+0.01);
                    obj.P{l}{n}=(obj.P{l}{n}-obj.G{l}{n}*obj.nn.Z{l-1}'*obj.P{l}{n})/0.01;                    
                    obj.nn.W{l}(n,:)=obj.nn.W{l}(n,:)+obj.G{l}{n}'*obj.D{l}(n);
                end
            end
        end
    end
    
end

