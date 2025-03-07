CREATE TABLE IF NOT EXISTS factura (
  idFactura number(10) NOT NULL AUTO_INCREMENT,
  revisionNumber  number(10) NOT NULL,
  buyDate TIMESTAMP NOT NULL,
  subtotal number(20,2),
  PRIMARY KEY(idFactura),
);