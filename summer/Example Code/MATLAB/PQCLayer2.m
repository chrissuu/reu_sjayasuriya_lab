classdef PQCLayer2 < nnet.layer.Layer
    % Custom PQC layer example.

    properties (Learnable)
        % Define layer learnable parameters.
        P1
        P2
        P3
        P4
    end

    methods
        function layer = PQCLayer2
            % Set layer name.
            layer.Name = "PQCLayer2";

            % Set layer description.
            layer.Description = "Layer containing a parameterized " + ...
                "quantum circuit (PQC)";

            % Initialize learnable parameter.
            layer.P1 = rand;
            layer.P2 = rand;
            layer.P3 = rand;
            layer.P4 = rand;
        end

        function Z = predict(layer,X)
            % Z = predict(layer,X) forwards the input data X through the
            % layer and outputs the result Z at prediction time.
             Z = computeZ(X, [layer.P1, layer.P2, layer.P3, layer.P4]);
        end

        function [dLdX,dLdP1,dLdP2,dLdP3,dLdP4] = backward(layer,X,Z,dLdZ,memory)
            % Backpropagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %     layer   - Layer though which data backpropagates
            %     X       - Layer input data
            %     Z       - Layer output data
            %     dLdZ    - Derivative of loss with respect to layer
            %               output
            %     memory  - Memory value from forward function
            % Outputs:
            %     dLdX   - Derivative of loss with respect to layer input
            %     dLdA   - Derivative of loss with respect to learnable
            %              parameter A
            %     dLdB   - Derivative of loss with respect to learnable
            %              parameter B

            s = pi/8;
            ZPlus  = computeZ(X, [layer.P1 + s, layer.P2, layer.P3, layer.P4]);
            ZMinus = computeZ(X, [layer.P1 - s, layer.P2, layer.P3, layer.P4]);
            dZdP1 = X(1,:).*(ZPlus-ZMinus);%X(1,:).*((ZPlus - ZMinus)./(2*sin(X(1,:).*s)));
            dLdP1 = sum(dLdZ.*dZdP1,"all");

            ZPlus  = computeZ(X, [layer.P1, layer.P2, layer.P3 + s, layer.P4]);
            ZMinus = computeZ(X,[layer.P1, layer.P2, layer.P3 - s, layer.P4]);
            dZdP3 = X(1,:).*(ZPlus-ZMinus);%X(1,:).*((ZPlus - ZMinus)./(2*sin(X(1,:).*s)));
            dLdP3 = sum(dLdZ.*dZdP3,"all");

            ZPlus  = computeZ(X, [layer.P1, layer.P2 + s, layer.P3, layer.P4]);
            ZMinus = computeZ(X,[layer.P1, layer.P2 - s, layer.P3, layer.P4]);
            dZdP2 = X(2,:).*(ZPlus-ZMinus);%X(2,:).*((ZPlus - ZMinus)./(2*sin(X(2,:).*s)));
            dLdP2 = sum(dLdZ.*dZdP2,"all");

            ZPlus  = computeZ(X, [layer.P1, layer.P2, layer.P3, layer.P4 + s]);
            ZMinus = computeZ(X,[layer.P1, layer.P2, layer.P3, layer.P4 - s]);
            dZdP4 = X(2,:).*(ZPlus-ZMinus);%X(2,:).*((ZPlus - ZMinus)./(2*cos(X(2,:).*s)));
            dLdP4 = sum(dLdZ.*dZdP4,"all");

            % Set the gradients with respect to x and y to zero
            % because the QNN does not use them during training.
            dLdX = zeros(size(X),"like",X);
        end
    end
end

function circ = ZZFeatureMap(X)
    t=0;
    p=0;
    l=2*X(1);
    u1_1 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];
    l=2*X(2);
    u1_2 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];
    l=2*(pi - X(1))*(pi-X(2));
    u1_3 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];

    gates = [hGate(1), hGate(2), unitaryGate(1,u1_1), unitaryGate(2, u1_2), ...
        cnotGate(1,2), unitaryGate(2,u1_3), cnotGate(1,2)];
    circ = quantumCircuit(gates,Name="ZZFeatureMap");
end

function circ = RealAmplitudes(P)
    gates = [ryGate(1,P(1)), ryGate(2,P(2)),  ...
        cnotGate(1,2), ryGate(1,P(3)), ryGate(2,P(4))];
    circ = quantumCircuit(gates,Name="RealAmplitudes");
end


function Z = computeZ(X, P)
    numSamples = size(X,2);
    x1 = X(1,:);
    x2 = X(2,:);
    Z = zeros(1,numSamples,"like",X);
    for j = 1:numSamples
        circ1 = ZZFeatureMap([x1(j), x2(j)]);
        circ2 = RealAmplitudes(P);
        totaGates = [compositeGate(circ1,[1 2])
              compositeGate(circ2,[1 2])];
        myCircuit = quantumCircuit(totaGates);
        s = simulate(myCircuit);
        Z(j) = probability(s,2,"0") - probability(s,2,"1");
    end
end